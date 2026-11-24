"""Leave-one-farm-out (treina numa fazenda, testa na outra) para os 4 modelos, com as configs de
`repeated_cv.py` (reexecutado em 2026-11-24, sem `media_suplemento_kg_dia`). MLP = média de 5 redes.
Saída: results/lofo_results.csv (uma linha por modelo e fazenda de teste, + preditor da média)."""
import os, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_prep as dp
import repeated_cv as rc

if __name__ == "__main__":
    from dotenv import load_dotenv; load_dotenv()
    d = dp.get_data(); df, cols = d["df"], d["scale_cols"]
    rows = []
    for tr, te, farm in dp.leave_one_property_out(d["groups"]):
        y = df.iloc[te][dp.TARGET].values
        naive = np.full(len(te), df.iloc[tr][dp.TARGET].mean())
        rows.append(dict(model="Media_treino", test_farm=farm, **dp.metrics(y, naive)))
        for m in rc.ML_MODELS:
            out = rc.predict(m, df.iloc[tr], df.iloc[te], cols, seed=dp.SEED, n_ens=5)
            rows.append(dict(model=m, test_farm=farm, **dp.metrics(y, out["pred"])))
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows); res.round(4).to_csv(f"{rc.OUT}/lofo_results.csv", index=False)
    print(res.round(4).to_string(index=False))
