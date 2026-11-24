"""Climatologia diaria de T2M (NASA POWER, API diaria, MERRA-2) no ponto das fazendas, media 2001-2019
por dia do ano (fora dos anos do estudo, 2020+). As duas fazendas caem na mesma celula da grade
(diferenca < 0,05 C entre as temperaturas do dataset e as recalculadas com as coordenadas da fazenda A).
Saidas: data/climatologia_t2m_2001_2019.csv (dia do ano, T2M medio)."""
import numpy as np, pandas as pd, requests
r = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point",
                 params=dict(parameters="T2M", community="AG", longitude=-46.0748, latitude=-20.0072,
                             start="20010101", end="20191231", format="JSON"), timeout=180)
t = pd.Series(r.json()["properties"]["parameter"]["T2M"]); t.index = pd.to_datetime(t.index, format="%Y%m%d")
t = t.replace(-999, np.nan)
doy = t.groupby([t.index.month, t.index.day]).mean()
doy.index = [f"{m:02d}-{d:02d}" for m, d in doy.index]
doy.rename("t2m_clim_c").to_csv("data/climatologia_t2m_2001_2019.csv", index_label="mes_dia")
print(len(doy), doy.round(2).head(3).to_dict())
