# dnn_cattle_weight_gain

## Visão geral

Rede neural densa (DNN/MLP em **PyTorch**) que prevê e classifica o **Ganho Médio Diário (GMD, `adg_kg_day`)** de bovinos de corte. Projeto acadêmico de TCC — IFMG, Depto. de Engenharia e Computação (Prof. Ciniro Nametala; aluno Euler Gomes). Modelo `GMDNN`, ainda em desenvolvimento.

## Estrutura

- `dnn_cattle_weight_gain.ipynb`, `cattle_gmd_dnn.ipynb` — notebooks de treino/análise (rodar todas as células).
- `data/` — datasets:
  - `cattle_dataset.csv` (247 amostras) — **dataset principal** já processado, 20 colunas (19 features + alvo `adg_kg_day`).
  - `cattle_dataset_2.csv` (494 amostras) — versão maior/alternativa, mesmo schema.
  - `cattle_gain.csv` (203 linhas) — dados brutos em português (schema antigo).
- `models/` — pesos treinados: `model_adg_dnn.pth`, `model_cattle_gain.pth`.
- `support_scripts/processar_formulario.py` — converte export do Google Forms → dataset pronto (idade, raça→% Bos indicus, forrageira→PB/digestibilidade, clima via NASA POWER API p/ Bambuí-MG, GMD). Uso: `python support_scripts/processar_formulario.py dados.csv [--sem-clima]`.
- `slides_tcc/` — apresentação LaTeX/Beamer do TCC. `monitoring.ods` — planilha de acompanhamento.

## Dados

- Coletados em duas fazendas na região de **Bambuí-MG**.
- Features de `cattle_dataset.csv` (ordem das colunas): `initial_weight_kg, age_days, sex_male, bos_indicus_proportion_pct, supplement_amount_kg_day, supplement_crude_protein_pct, supplement_metabolizable_energy, supplementation_frequency_days_week, forage_crude_protein_pct, forage_digestibility_pct, days_on_pasture, paddock_rotation, transport_stress, mean_temperature_c, accumulated_rainfall_mm, days_since_health_event, health_event_duration_days, recent_vaccination, recent_deworming` → alvo `adg_kg_day`.

## Ambiente

- Python 3.11; PyTorch 2.10 + CUDA 13; NumPy, Pandas, Scikit-learn, Matplotlib/Seaborn, pygame, torchinfo.
- Setup: `conda create -n cattle_env python=3.11 && pip install -r requirements.txt`.

---

## Onde paramos

**Última atualização:** 2026-07-11

**Tópico atual:** Verificação da integridade do arquivo `data/cattle_dataset.csv` (247 amostras, 20 colunas).

### Progresso na verificação de integridade (coluna a coluna)
- ✅ `supplement_amount_kg_day` (col 5): outliers corrigidos pelo usuário (linhas 16, 37 e 151 — esta última 147,0 → 1,47). Agora todos entre 0,248 e 1,980; média 0,93, dp 0,34.

### Problemas ainda pendentes (a investigar/confirmar)
- `adg_kg_day` negativo em 4 linhas (mínimo −0,70 kg/dia) — pode ser perda de peso real ou erro.
- `initial_weight_kg` mínimo de 25 kg — muito baixo para bovino, possível outlier/erro.

### Próximo passo
Continuar a verificação coluna a coluna e decidir o que fazer com o adg negativo e o peso de 25 kg.

---

> **Nota de trabalho:** manter esta seção "Onde paramos" atualizada ao final de cada sessão (data, tópico atual, próximo passo) e retomá-la no início da próxima.
