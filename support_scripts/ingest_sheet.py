"""Ingestão de folha de campo manuscrita -> data/dataset_por_animal_modelo_v3.csv.

Fluxo: transcreve-se uma folha (1 linha = 1 animal), informam-se as constantes
da fazenda, e este módulo valida a consistência, deriva o clima (NASA POWER) e
ANEXA as linhas ao dataset por-animal.

Uso (a partir da raiz do repositório):

    import sys; sys.path.append("support_scripts")
    import ingest_sheet as ing

    fazenda = dict(
        id_propriedade="elvis",
        lat=-20.0072, lon=-46.0748,          # graus decimais (S/W negativos)
        sexo_macho=0,                        # 0=fêmea, 1=macho (constante da folha)
        proporcao_bos_indicus_pct=75,
        pb_suplemento_pct=30,
        frequencia_suplementacao_dias_semana=7,
        media_proteina_bruta_forragem_pct=10.42,
        rotacao_piquete=1,
        numero_eventos_transporte=1,
        suplemento_frac_peso=0.003,          # 0,3% do peso médio do ciclo
    )
    animais = [
        # id como STRING (preserva zero à esquerda). gmd_folha/ganho_folha são
        # opcionais e servem só p/ conferir a leitura contra a folha.
        dict(id="401", entrada=154, saida=276, dias=338, venda="2025-01-14",
             ganho_folha=122, gmd_folha=0.360),
        ...
    ]

    ing.ingest(animais, fazenda, dry_run=True)   # só valida e mostra
    ing.ingest(animais, fazenda)                 # valida, deriva clima e ANEXA

Convenção de datas: `data_saida` = data de venda; `data_entrada` = venda − dias
(o `dias` da folha, validado pelo gmd, é mais confiável que a compra manuscrita).
Para forçar uma entrada específica, passe `entrada_data="YYYY-MM-DD"` na linha.
"""
import json
import urllib.request
from datetime import date, timedelta

import pandas as pd

DATA_PATH = "data/dataset_por_animal_modelo_v3.csv"
DRY_MONTHS = {4, 5, 6, 7, 8, 9}        # estação seca (abr–set) p/ proporcao_ciclo_seca
GMD_TOL = 0.0016                       # tolerância p/ conferir gmd_folha (arredondamento)

# ordem/nomes das colunas do schema por-animal (v3, 19 colunas)
SCHEMA = [
    "id_animal", "id_propriedade", "data_entrada", "data_saida",
    "peso_entrada_kg", "peso_saida_kg", "dias_permanencia", "gmd_kg_dia",
    "sexo_macho", "proporcao_bos_indicus_pct", "media_suplemento_kg_dia",
    "pb_suplemento_pct", "frequencia_suplementacao_dias_semana",
    "media_proteina_bruta_forragem_pct", "rotacao_piquete",
    "numero_eventos_transporte", "temperatura_media_c",
    "precipitacao_acumulada_mm", "proporcao_ciclo_seca",
]
FARM_COLS = [
    "id_propriedade", "sexo_macho", "proporcao_bos_indicus_pct", "pb_suplemento_pct",
    "frequencia_suplementacao_dias_semana", "media_proteina_bruta_forragem_pct",
    "rotacao_piquete", "numero_eventos_transporte",
]


def nasa_power(lat, lon, start, end):
    """Baixa T2M (média diária) e PRECTOTCORR (chuva diária) do NASA POWER."""
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point"
           f"?parameters=T2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}"
           f"&start={start}&end={end}&format=JSON")
    p = json.load(urllib.request.urlopen(url, timeout=120))["properties"]["parameter"]
    return p["T2M"], p["PRECTOTCORR"]


def _entrada_de(a):
    if a.get("entrada_data"):
        return date.fromisoformat(a["entrada_data"])
    return date.fromisoformat(a["venda"]) - timedelta(int(a["dias"]))


def ingest(animais, fazenda, csv_path=DATA_PATH, dry_run=False):
    """Valida, deriva clima e anexa as linhas. Retorna o DataFrame das novas linhas.

    Levanta ValueError se houver erro grave (ganho_folha divergente, dias<=0 ou
    brinco já existente). Divergência de gmd_folha é só aviso.
    """
    for k in FARM_COLS + ["lat", "lon", "suplemento_frac_peso"]:
        if k not in fazenda:
            raise ValueError(f"faltou a chave '{k}' em `fazenda`")

    # datas -> uma única consulta ao NASA POWER cobrindo tudo
    entradas = {a["id"]: _entrada_de(a) for a in animais}
    start = min(entradas.values()).strftime("%Y%m%d")
    end = max(date.fromisoformat(a["venda"]) for a in animais).strftime("%Y%m%d")
    T2M, PREC = nasa_power(fazenda["lat"], fazenda["lon"], start, end)

    df = pd.read_csv(csv_path, dtype={"id_animal": str}) if _exists(csv_path) else None
    existentes = set(df["id_animal"]) if df is not None else set()
    cols = list(df.columns) if df is not None else SCHEMA
    frac = fazenda["suplemento_frac_peso"]

    novos, erros = [], []
    print(f"{'brinco':8s} {'dias':>4s} {'gmd':>7s} {'sup':>6s} {'temp':>6s} "
          f"{'chuva':>7s} {'seca':>6s}  obs")
    for a in animais:
        aid = str(a["id"])
        ent_p, sai_p, dias = int(a["entrada"]), int(a["saida"]), int(a["dias"])
        entrada = entradas[aid]
        obs = []
        if dias <= 0:
            erros.append(f"{aid}: dias<=0")
        ganho = sai_p - ent_p
        if "ganho_folha" in a and a["ganho_folha"] != ganho:
            erros.append(f"{aid}: ganho {ganho} != folha {a['ganho_folha']}")
            obs.append("GANHO!")
        gmd = ganho / dias if dias else float("nan")
        if "gmd_folha" in a and abs(gmd - a["gmd_folha"]) > GMD_TOL:
            obs.append(f"gmd folha={a['gmd_folha']}?")
        if aid in existentes or aid in {str(n['id_animal']) for n in novos}:
            erros.append(f"{aid}: brinco duplicado")
            obs.append("DUP!")

        days = [entrada + timedelta(d) for d in range(dias)]
        keys = [d.strftime("%Y%m%d") for d in days]
        temps = [T2M[k] for k in keys if T2M.get(k, -999.0) != -999.0]
        precs = [PREC[k] for k in keys if PREC.get(k, -999.0) != -999.0]
        row = {c: fazenda[c] for c in FARM_COLS}
        row.update(
            id_animal=aid, data_entrada=entrada.isoformat(), data_saida=a["venda"],
            peso_entrada_kg=ent_p, peso_saida_kg=sai_p, dias_permanencia=dias,
            gmd_kg_dia=round(gmd, 6),
            media_suplemento_kg_dia=round(frac * (ent_p + sai_p) / 2, 3),
            temperatura_media_c=round(sum(temps) / len(temps), 2),
            precipitacao_acumulada_mm=round(sum(precs), 1),
            proporcao_ciclo_seca=round(sum(1 for d in days if d.month in DRY_MONTHS) / dias, 4),
        )
        novos.append(row)
        print(f"{aid:8s} {dias:4d} {gmd:7.3f} {row['media_suplemento_kg_dia']:6.3f} "
              f"{row['temperatura_media_c']:6.2f} {row['precipitacao_acumulada_mm']:7.1f} "
              f"{row['proporcao_ciclo_seca']:6.3f}  {' '.join(obs)}")

    new_df = pd.DataFrame(novos)[cols]
    if erros:
        raise ValueError("ERROS (nada foi escrito):\n  - " + "\n  - ".join(erros))
    if dry_run:
        print(f"\n[dry-run] {len(new_df)} linhas validadas, NADA escrito.")
        return new_df

    out = pd.concat([df, new_df], ignore_index=True) if df is not None else new_df
    out.to_csv(csv_path, index=False)
    print(f"\nAnexadas {len(new_df)} linhas. Total no arquivo: {len(out)}.")
    return new_df


def _exists(path):
    import os
    return os.path.exists(path)
