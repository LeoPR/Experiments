#!/usr/bin/env python3
"""
infer.py - Interface de inferência e plot (refatorado).

Agora delega projeção ao model.py (load_model / project) e centraliza apenas
salvamento de CSV e geração de gráficos.

Comportamento idêntico ao anterior: nomes de arquivos, corte de janela,
formatação de rótulos com 'h' e exemplos em __main__ mantidos.
"""

import os
import json
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Importar as funções/constantes do model.py
from model import (
    load_model,
    project,
    _load_observations,
    slugify,
    CFG,
    FORECAST_DIR,
    IMAGES_DIR,
    utcnow_z
)

def save_forecast_csv(forecast_df, equipment_eff, horizon, resolution):
    os.makedirs(FORECAST_DIR, exist_ok=True)
    fname = f"{slugify(equipment_eff or 'general')}_forecast_{horizon}_{resolution}.csv.gz"
    out_path = os.path.join(FORECAST_DIR, fname)
    forecast_df.to_csv(out_path, index=False, compression="gzip")
    print(f"[{utcnow_z()}] Forecast salvo: {out_path}")
    return out_path

def plot_forecast(forecast_df, obs, equipment_eff, meta, horizon, resolution, save_image=True, image_name=None):
    """
    Plota o forecast (histórico + futuro) com a mesma estética e lógica do infer.py original.
    - forecast_df: DataFrame com colunas ['ds','yhat','yhat_lower','yhat_upper']
    - obs: DataFrame com colunas ['ds','y'] (já filtrado para a janela de exibição)
    - meta: dicionário retornado por project() contendo display_start/display_end/resolution_norm, etc.
    """
    plt.figure(figsize=(11, 6))
    ax = plt.gca()

    # garantir tipos
    if not forecast_df.empty and not pd.api.types.is_datetime64_any_dtype(forecast_df['ds']):
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    if not obs.empty and not pd.api.types.is_datetime64_any_dtype(obs['ds']):
        obs['ds'] = pd.to_datetime(obs['ds'])

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

    # Formatação dos rótulos de hora com sufixo "h" (mesma lógica de antes)
    try:
        display_start = meta.get("display_start")
        display_end = meta.get("display_end")
        resolution_norm = meta.get("resolution_norm", "").lower()
        display_span = (display_end - display_start) if (display_start is not None and display_end is not None) else None
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
    return

def infer(equipment_name=None, start=None, horizon="1H", resolution="15min", save_image=True, image_name=None):
    """
    Fluxo de inferência refatorado:
      1) carrega observados (mantendo extras e auto-detect)
      2) carrega modelo (específico se existir, senão geral)
      3) calcula start_ts (prioridade start -> last_ds_model -> max obs -> now)
      4) chama project() para obter forecast_df e obs filtrado
      5) salva CSV e plota (com plot_forecast)
    """
    # 1) carregar observados e possivelmente detectar equipamento
    obs, equipment_eff = _load_observations(equipment_name)

    # 2) carregar modelo
    model_obj, model_path, last_ds = load_model(equipment_eff)

    # 3) decidir start_ts
    if start:
        start_ts = pd.to_datetime(start)
    else:
        start_ts = last_ds if last_ds is not None else (obs["ds"].max() if not obs.empty else pd.Timestamp.now())

    # 4) projeção (usa project)
    forecast_df, obs_filtered, meta = project(model_obj, start_ts, horizon, resolution, obs_df=obs)

    # 5) salvar CSV e plotar
    save_forecast_csv(forecast_df, equipment_eff, horizon, resolution)
    plot_forecast(forecast_df, obs_filtered, equipment_eff, meta, horizon, resolution, save_image=save_image, image_name=image_name)

    return forecast_df

# exemplos de uso (mantidos, idênticos ao anterior)
if __name__ == "__main__":
    infer_cfg = CFG.get("infer", {})
    examples = infer_cfg.get("examples", {})

    try:
        e1 = examples.get("hour", {"horizon": "1H", "resolution": "15min"})
        infer(equipment_name=None, start=None, horizon=e1["horizon"], resolution=e1["resolution"], save_image=True)
    except Exception as err:
        print("Erro exemplo 1:", err)

    try:
        e2 = examples.get("week", {"horizon": "7D", "resolution": "H"})
        infer(equipment_name=None, start=None, horizon=e2["horizon"], resolution=e2["resolution"], save_image=True)
    except Exception as err:
        print("Erro exemplo 2:", err)

    try:
        e3 = examples.get("month", {"horizon": "30D", "resolution": "D"})
        infer(equipment_name=None, start=None, horizon=e3["horizon"], resolution=e3["resolution"], save_image=True)
    except Exception as err:
        print("Erro exemplo 3:", err)