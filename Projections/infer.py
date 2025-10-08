#!/usr/bin/env python3
"""
infer.py - Usa o núcleo em model.py para gerar forecast e plotar (gráfico + CSV, comportamento idêntico ao anterior).

Agora a lógica de carregamento de dados, modelo e predição está em model.py (função generate_forecast).
Este arquivo:
  - Chama generate_forecast(...)
  - Salva CSV comprimido no diretório de forecasts
  - Gera gráfico com as mesmas regras (IC histórico/futuro, cores, rótulos de hora com 'h')
  - Mantém o bloco de exemplos no __main__

Uso:
  pip install pandas prophet joblib matplotlib
  python infer.py
"""

import os
import json
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from model import (
    generate_forecast,
    slugify,
    CFG,
    FORECAST_DIR,
    IMAGES_DIR
)

def utcnow_z():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def infer(equipment_name=None, start=None, horizon="1H", resolution="15min", save_image=True, image_name=None):
    """
    Mantém a mesma assinatura anterior.
    Agora usa generate_forecast do model.py.
    """
    forecast_df, obs, equipment_eff, meta = generate_forecast(
        equipment_name=equipment_name,
        start=start,
        horizon=horizon,
        resolution=resolution
    )

    # Garantir diretório de forecast e salvar CSV (parte filtrada, como antes)
    os.makedirs(FORECAST_DIR, exist_ok=True)
    fname = f"{slugify(equipment_eff or 'general')}_forecast_{horizon}_{resolution}.csv.gz"
    out_path = os.path.join(FORECAST_DIR, fname)
    forecast_df.to_csv(out_path, index=False, compression="gzip")
    print(f"[{utcnow_z()}] Forecast salvo: {out_path}")

    # Plot (idêntico à versão anterior)
    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    last_hist_date = obs['ds'].max() if not obs.empty else forecast_df['ds'].min()
    is_future = forecast_df['ds'] > last_hist_date
    is_past = ~is_future

    if is_past.any():
        plt.fill_between(forecast_df.loc[is_past, 'ds'],
                         forecast_df.loc[is_past, 'yhat_lower'],
                         forecast_df.loc[is_past, 'yhat_upper'],
                         color='lightgray', alpha=0.4, label='IC (histórico)')
    if is_future.any():
        plt.fill_between(forecast_df.loc[is_future, 'ds'],
                         forecast_df.loc[is_future, 'yhat_lower'],
                         forecast_df.loc[is_future, 'yhat_upper'],
                         color='salmon', alpha=0.35, label='IC (futuro)')

    if not obs.empty:
        plt.scatter(obs['ds'], obs['y'], color='black', s=18, label='Observado')

    plt.plot(forecast_df['ds'], forecast_df['yhat'], color='lightblue', linewidth=1, label='yhat')
    if is_future.any():
        plt.plot(forecast_df.loc[is_future, 'ds'],
                 forecast_df.loc[is_future, 'yhat'],
                 color='red', linewidth=2, label='Futuro')

    # Formatação de rótulos de hora
    try:
        display_start = meta.get("display_start")
        display_end = meta.get("display_end")
        display_span = (display_end - display_start) if (display_start and display_end) else None
        resolution_norm = meta.get("resolution_norm", "").lower()
        short_window = (display_span is not None and display_span <= pd.Timedelta("2d"))
        is_hour_like = ("h" in resolution_norm and "min" not in resolution_norm) or ("min" in resolution_norm) or (resolution_norm in ("t", "min"))
        if short_window and is_hour_like:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Hh"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()
    except Exception:
        pass

    plt.xlabel("Data")
    plt.ylabel("Value")
    plt.title(f"Forecast - {equipment_eff or 'general'} - horizon {horizon} res {resolution}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_image:
        os.makedirs(IMAGES_DIR, exist_ok=True)
        img_name = image_name or f"forecast_{slugify(equipment_eff or 'general')}_{horizon}_{resolution}.png"
        img_path = os.path.join(IMAGES_DIR, img_name)
        plt.savefig(img_path)
        print(f"[{utcnow_z()}] Gráfico salvo: {img_path}")

    plt.show()
    return forecast_df

# Bloco de exemplos (mesmo padrão anterior, apenas chamando a nova função infer)
if __name__ == "__main__":
    infer_cfg = CFG.get("infer", {})
    examples = infer_cfg.get("examples", {})
    # Exemplo hora
    try:
        e1 = examples.get("hour", {"horizon": "1H", "resolution": "15min"})
        infer(equipment_name=None, start=None, horizon=e1["horizon"], resolution=e1["resolution"], save_image=True)
    except Exception as err:
        print("Erro exemplo 1:", err)
    # Exemplo semana
    try:
        e2 = examples.get("week", {"horizon": "7D", "resolution": "H"})
        infer(equipment_name=None, start=None, horizon=e2["horizon"], resolution=e2["resolution"], save_image=True)
    except Exception as err:
        print("Erro exemplo 2:", err)
    # Exemplo mês
    try:
        e3 = examples.get("month", {"horizon": "30D", "resolution": "D"})
        infer(equipment_name=None, start=None, horizon=e3["horizon"], resolution=e3["resolution"], save_image=True)
    except Exception as err:
        print("Erro exemplo 3:", err)