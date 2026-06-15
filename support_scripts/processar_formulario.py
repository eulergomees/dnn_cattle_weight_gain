"""
processar_formulario.py
========================
Converte as respostas do Google Forms (exportadas como CSV via Google Sheets)
no dataset final com as 20 variáveis para o modelo MLP.

Transformações automáticas:
  - Data de nascimento + data da pesagem  → idade_dias
  - Dropdown de raça                      → proporcao_bos_indicus_pct
  - Dropdown de forrageira + estação      → proteina_bruta_forragem_pct
                                            digestibilidade_forragem_pct
  - Data de entrada + data da pesagem     → dias_permanecia
  - NASA POWER API (período de avaliação) → temperatura_media_c
                                            precipitacao_acumulada_mm
  - Data do evento sanitário              → dias_desde_evento_sanitario
  - Peso de saída - peso inicial / dias   → saida_gmd_kg_dia

Uso:
  python processar_formulario.py respostas.csv
  python processar_formulario.py respostas.csv --sem-clima   # sem buscar clima

O arquivo de saída será: dataset_gmd_YYYYMMDD.csv
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO — ajuste conforme necessário
# ══════════════════════════════════════════════════════════════════════

LATITUDE = -20.05   # Coordenadas de Bambuí-MG
LONGITUDE = -45.97  # Ajuste para a propriedade específica

# ── Mapeamento: raça → % Bos indicus ────────────────────────────────

BREED_MAP = {
    "Nelore":                          100.0,
    "Tabapuã":                         100.0,
    "Guzerá":                          100.0,
    "Brahman":                         100.0,
    "Gir":                             100.0,
    "½ sangue (F1 Taurino × Zebuíno)":  50.0,
    "¾ Zebuíno":                        75.0,
    "¾ Taurino":                        25.0,
    "Angus":                             0.0,
    "Senepol":                           0.0,
    "Charolês":                          0.0,
    "Hereford":                          0.0,
}

# ── Lookup CQBAL: forrageira → (PB águas, PB seca, Dig águas, Dig seca) ─

FORAGE_MAP = {
    "Marandu":   (9.5,  5.8, 58.0, 48.0),
    "MG-4":      (9.5,  5.8, 58.0, 48.0),
    "MG-5 / Xaraés": (10.5, 6.5, 60.0, 50.0),
    "Mombaça":   (11.5, 6.8, 60.0, 49.0),
    "Tanzânia":  (11.0, 6.5, 56.0, 47.0),
    "Tifton":    (13.0, 9.0, 62.0, 54.0),
    "Estrela":   (11.0, 7.0, 56.0, 47.0),
    "Decumbens": (8.5,  4.8, 55.0, 45.0),
}

# ── Nomes das colunas no Google Forms (devem bater exatamente) ──────

COL = {
    "data_pesagem":         "Data da ultima pesagem",
    "peso_inicial":         "Peso inicial (kg)",
    "data_nascimento":      "Data de nascimento",
    "sexo":                 "Sexo",
    "raca":                 "Composição racial",
    "forrageira":           "Espécies forrageiras do piquete",
    "supl_qtd":             "Quantidade de suplemento (kg/dia)",
    "supl_pb":              "PB do suplemento (% MS)",
    "supl_em":              "Energia metabolizável do suplemento (Mcal/kg MS)",
    "supl_freq":            "Frequência de suplementação (dias/semana)",
    "data_entrada":         "Data de entrada no lote",
    "rotacao":              "Rotação de piquete",
    "estresse":             "Estresse de transporte (últimos 7 dias da ultima pesagem)",
    "evento_sanitario":     "Evento sanitário recente",
    "data_evento":          "Data do evento sanitário",
    "duracao_evento":       "Duração do evento (dias)",
    "vacina":               "Vacinação recente (últimos 7 dias)",
    "vermifugacao":         "Vermifugação recente (últimos 7 dias)",
    "peso_saida":           "Peso de saída (kg)",
}


# ══════════════════════════════════════════════════════════════════════
#  FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════════

def is_aguas(dt: datetime) -> bool:
    """Retorna True se a data cai no período de águas (out–mar)."""
    return dt.month >= 10 or dt.month <= 3


def parse_date(val) -> pd.Timestamp | None:
    """Tenta parsear data em vários formatos comuns."""
    if pd.isna(val):
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return pd.Timestamp(datetime.strptime(str(val).strip(), fmt))
        except ValueError:
            continue
    return pd.to_datetime(val, errors="coerce")


def sim_nao_to_int(val) -> int:
    """Converte 'Sim'/'Não' para 1/0."""
    if pd.isna(val):
        return 0
    return 1 if str(val).strip().lower() in ("sim", "yes", "1", "true") else 0


def fetch_clima(data_ini: datetime, data_fim: datetime,
                lat: float = LATITUDE,
                lon: float = LONGITUDE) -> tuple[float | None, float | None]:
    """
    Busca temperatura média e precipitação acumulada via NASA POWER API.
    https://power.larc.nasa.gov/docs/services/api/temporal/daily/

    Parâmetros:
      T2M          — Temperatura a 2m (°C), média diária
      PRECTOTCORR  — Precipitação corrigida (mm/dia)

    Retorna (temp_media, precip_acum) ou (None, None) em caso de falha.
    """
    try:
        import requests
    except ImportError:
        print("  ⚠ Instale 'requests': pip install requests")
        return None, None

    ini = data_ini.strftime("%Y%m%d")
    fim = data_fim.strftime("%Y%m%d")

    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,PRECTOTCORR"
        f"&community=AG"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={ini}"
        f"&end={fim}"
        f"&format=JSON"
    )

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None, None

        data = resp.json()
        params = data.get("properties", {}).get("parameter", {})
        t2m = params.get("T2M", {})
        prec = params.get("PRECTOTCORR", {})

        if not t2m:
            return None, None

        # Filtrar valores inválidos (NASA POWER usa -999 para missing)
        temps = [v for v in t2m.values() if v > -900]
        precips = [v for v in prec.values() if v >= 0]

        if not temps:
            return None, None

        temp_media = round(sum(temps) / len(temps), 1)
        precip_acum = round(sum(precips), 1)

        return temp_media, precip_acum

    except Exception as e:
        print(f"  ⚠ Erro ao buscar NASA POWER: {e}")
        return None, None


# ══════════════════════════════════════════════════════════════════════
#  PROCESSAMENTO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def processar(csv_path: str, buscar_clima: bool = True) -> pd.DataFrame:
    """Lê o CSV do Google Forms e retorna o DataFrame com as 20 variáveis."""

    print(f"📂 Lendo: {csv_path}")
    df = pd.read_csv(csv_path)

    # Verificar colunas presentes
    # Campos opcionais que podem não existir no Forms
    OPTIONAL_COLS = {"data_evento"}
    colunas_esperadas = [
        v for k, v in COL.items() if k not in OPTIONAL_COLS
    ]
    faltando = [c for c in colunas_esperadas if c not in df.columns]
    if faltando:
        print(f"\n⚠ Colunas não encontradas no CSV: {faltando}")
        print("  Verifique se os títulos das perguntas no Google Forms "
              "batem exatamente com a configuração COL neste script.")
        print(f"\n  Colunas disponíveis no CSV:\n    {list(df.columns)}")
        sys.exit(1)

    has_data_evento = COL["data_evento"] in df.columns
    if not has_data_evento:
        print("  ⚠ Coluna 'Data do evento sanitário' ausente — "
              "dias_desde_evento_sanitario será calculado pela duração.")

    print(f"📊 {len(df)} respostas encontradas.\n")

    rows = []
    inmet_cache: dict[str, tuple] = {}  # cache por período

    for idx, r in df.iterrows():
        print(f"  Processando amostra {idx + 1}/{len(df)}...", end="")

        # ── Datas ──
        dt_pesagem = parse_date(r[COL["data_pesagem"]])
        dt_nascimento = parse_date(r[COL["data_nascimento"]])
        dt_entrada = parse_date(r[COL["data_entrada"]])
        dt_evento = (
            parse_date(r[COL["data_evento"]])
            if has_data_evento and COL["data_evento"] in r
            else None
        )

        if dt_pesagem is None:
            print(" ✗ data de pesagem inválida, pulando.")
            continue

        # ── Variáveis calculadas ──
        idade_dias = (
            (dt_pesagem - dt_nascimento).days if dt_nascimento else 0
        )
        dias_permanencia = (
            (dt_pesagem - dt_entrada).days if dt_entrada else 0
        )
        dias_desde_evento = 0
        if sim_nao_to_int(r[COL["evento_sanitario"]]):
            if dt_evento:
                dias_desde_evento = (dt_pesagem - dt_evento).days
            else:
                # Sem data do evento: usa duração como proxy
                dur = int(float(r[COL["duracao_evento"]] or 0))
                dias_desde_evento = dur if dur > 0 else 1

        # ── Raça → % Bos indicus ──
        raca = str(r[COL["raca"]]).strip()
        prop_indicus = BREED_MAP.get(raca, 100.0)

        # ── Forrageira → PB e Digestibilidade (média simples das espécies) ──
        forrageiras_raw = str(r[COL["forrageira"]]).strip()
        # Google Forms separa múltiplas seleções com "; " no CSV
        especies = [f.strip() for f in forrageiras_raw.split(";") if f.strip()]
        aguas = is_aguas(dt_pesagem)

        pbs, digs = [], []
        for esp in especies:
            dados = FORAGE_MAP.get(esp)
            if dados is None:
                # Tentar match parcial (ex: "Braquiária" genérica → Marandu)
                dados = FORAGE_MAP.get("Marandu")
            if dados:
                pbs.append(dados[0] if aguas else dados[1])
                digs.append(dados[2] if aguas else dados[3])

        # Fallback: se nenhuma espécie reconhecida, usa Marandu
        if not pbs:
            fallback = FORAGE_MAP["Marandu"]
            pbs = [fallback[0] if aguas else fallback[1]]
            digs = [fallback[2] if aguas else fallback[3]]

        pb_forragem = round(sum(pbs) / len(pbs), 1)
        dig_forragem = round(sum(digs) / len(digs), 1)

        # ── Sexo ──
        sexo_macho = 1 if str(r[COL["sexo"]]).strip().lower() == "macho" else 0

        # ── Numéricos diretos ──
        def safe_float(val, default=0.0):
            if pd.isna(val) or val == "":
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        peso_ini = safe_float(r[COL["peso_inicial"]])
        peso_saida = safe_float(r[COL["peso_saida"]])
        supl_qtd = safe_float(r[COL["supl_qtd"]])
        supl_pb = safe_float(r[COL["supl_pb"]])
        supl_em = safe_float(r[COL["supl_em"]])
        supl_freq = int(safe_float(r[COL["supl_freq"]]))
        duracao_evt = int(safe_float(r[COL["duracao_evento"]]))

        # ── Binários ──
        rotacao = sim_nao_to_int(r[COL["rotacao"]])
        estresse = sim_nao_to_int(r[COL["estresse"]])
        vacina = sim_nao_to_int(r[COL["vacina"]])
        vermifugacao = sim_nao_to_int(r[COL["vermifugacao"]])

        # ── Clima (NASA POWER) ──
        temp_media = None
        precip_acum = None

        if buscar_clima and dt_entrada and dt_pesagem:
            cache_key = f"{dt_entrada.strftime('%Y%m%d')}_{dt_pesagem.strftime('%Y%m%d')}"
            if cache_key in inmet_cache:
                temp_media, precip_acum = inmet_cache[cache_key]
            else:
                temp_media, precip_acum = fetch_clima(
                    dt_entrada, dt_pesagem
                )
                inmet_cache[cache_key] = (temp_media, precip_acum)

        # ── GMD ──
        gmd = (
            round((peso_saida - peso_ini) / dias_permanencia, 4)
            if dias_permanencia > 0 else 0.0
        )

        rows.append({
            "peso_inicial_kg": peso_ini,
            "idade_dias": idade_dias,
            "sexo_macho": sexo_macho,
            "proporcao_bos_indicus_pct": prop_indicus,
            "quantidade_suplemento_kg_dia": supl_qtd,
            "pb_suplemento_pct": supl_pb,
            "energia_metabolizavel_suplemento": supl_em,
            "frequencia_suplementacao_dias_semana": supl_freq,
            "proteina_bruta_forragem_pct": pb_forragem,
            "digestibilidade_forragem_pct": dig_forragem,
            "dias_permanecia": dias_permanencia,
            "rotacao_piquete": rotacao,
            "estresse_transporte": estresse,
            "temperatura_media_c": temp_media if temp_media else 0,
            "precipitacao_acumulada_mm": precip_acum if precip_acum else 0,
            "dias_desde_evento_sanitario": dias_desde_evento,
            "duracao_evento_sanitario_dias": duracao_evt,
            "vacina_recente": vacina,
            "vermifugacao_recente": vermifugacao,
            "saida_gmd_kg_dia": gmd,
        })
        print(" ✓")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Processa respostas do Google Forms → dataset GMD"
    )
    parser.add_argument("csv", help="Caminho do CSV exportado do Google Sheets")
    parser.add_argument(
        "--sem-clima", action="store_true",
        help="Não buscar dados climáticos na NASA POWER API"
    )
    parser.add_argument(
        "--saida", default=None,
        help="Nome do arquivo de saída (padrão: dataset_gmd_YYYYMMDD.csv)"
    )

    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"❌ Arquivo não encontrado: {args.csv}")
        sys.exit(1)

    dataset = processar(
        args.csv,
        buscar_clima=not args.sem_clima,
    )

    if dataset.empty:
        print("\n❌ Nenhuma amostra válida processada.")
        sys.exit(1)

    saida = args.saida or f"dataset_gmd_{datetime.now().strftime('%Y%m%d')}.csv"
    dataset.to_csv(saida, index=False)

    print(f"\n✅ Dataset salvo: {saida}")
    print(f"   {len(dataset)} amostras × {len(dataset.columns)} variáveis")

    # Resumo
    if "saida_gmd_kg_dia" in dataset.columns:
        gmd = dataset["saida_gmd_kg_dia"]
        print(f"\n📈 GMD — min: {gmd.min():.3f}  "
              f"média: {gmd.mean():.3f}  "
              f"max: {gmd.max():.3f} kg/dia")

    # Verificar campos sem dados climáticos
    sem_clima = dataset[
        (dataset["temperatura_media_c"] == 0) |
        (dataset["precipitacao_acumulada_mm"] == 0)
    ]
    if not sem_clima.empty:
        print(f"\n⚠ {len(sem_clima)} amostras sem dados climáticos. "
              "Preencha manualmente ou re-rode sem --sem-clima.")


if __name__ == "__main__":
    main()
