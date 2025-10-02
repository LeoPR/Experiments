# forecast.py
# Script principal que usa classes em projections.py e gera 3 PNGs (1h/1w/1m)
# Agora lê os parâmetros de horizonte a partir de horizons.json.
#
# Uso: coloque projections.py, config.json e horizons.json no mesmo diretório;
# abra no Spyder e execute. Troque MODEL no topo para 'prophet'|'arima'|'holt'.
#
# Requisitos: pandas, matplotlib, prophet (se usar prophet), statsmodels (se usar arima/holt).

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from projections import ProphetProjection, ARIMAProjection, HoltProjection

# ----- CONFIGURAÇÕES -----
MODEL = 'prophet'   # 'prophet', 'arima' ou 'holt'
INPUT_CSV = './dados/ME02__Medidor_Energy_P9-QDFI_VARIABLE_ENERGY_DEMAND_202510011009.csv'
CONFIG_FILE = 'config.json'
HORIZONS_FILE = 'horizons.json'
# -------------------------

# Carregar config do modelo
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
model_cfg = cfg.get(MODEL, {})

# Carregar horizontes
with open(HORIZONS_FILE, 'r', encoding='utf-8') as f:
    horizons_cfg = json.load(f)

# Carrega CSV (apenas Date e Value)
df = pd.read_csv(INPUT_CSV, parse_dates=['Date'])
df = df[['Date', 'Value']].rename(columns={'Date': 'ds', 'Value': 'y'})
df = df.sort_values('ds')
df['y'] = pd.to_numeric(df['y'], errors='coerce')

# Função de plot (mesma estética aprovada)
def plot_standard(df_obs, forecast, horizon_label, model_name, out_png):
    last_date = df_obs['ds'].max()
    is_future = forecast['ds'] > last_date
    is_past = ~is_future

    plt.figure(figsize=(11,6))

    if is_past.any():
        plt.fill_between(np.asarray(forecast.loc[is_past, 'ds'].dt.to_pydatetime()),
                         forecast.loc[is_past, 'yhat_lower'],
                         forecast.loc[is_past, 'yhat_upper'],
                         color='lightgray', alpha=0.4, label='IC (histórico)')

    if is_future.any():
        plt.fill_between(np.asarray(forecast.loc[is_future, 'ds'].dt.to_pydatetime()),
                         forecast.loc[is_future, 'yhat_lower'],
                         forecast.loc[is_future, 'yhat_upper'],
                         color='salmon', alpha=0.35, label='IC (futuro)')

    plt.scatter(df_obs['ds'], df_obs['y'], color='black', s=18, label='Observado')
    plt.plot(forecast['ds'], forecast['yhat'], color='lightblue', linestyle='-', linewidth=1, label='Previsão (yhat)')

    if is_future.any():
        plt.plot(forecast.loc[is_future, 'ds'], forecast.loc[is_future, 'yhat'], color='red', linewidth=2, label='Previsão futura')

    plt.xlabel('Data')
    plt.ylabel('Value')
    plt.title(f'Previsão - horizonte {horizon_label} ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Salvo: {out_png}")
    plt.show()


# Instanciar a projeção escolhida (carrega parâmetros do config.json)
train_mode = model_cfg.get('train_mode', 'all')
train_fraction = model_cfg.get('train_fraction', 0.8)
train_end = model_cfg.get('train_end', None)

if MODEL == 'prophet':
    proj = ProphetProjection(params=model_cfg.get('params', {}), train_mode=train_mode,
                             train_fraction=train_fraction, train_end=train_end)
elif MODEL == 'arima':
    proj = ARIMAProjection(params=model_cfg.get('params', {}), train_mode=train_mode,
                           train_fraction=train_fraction, train_end=train_end)
elif MODEL == 'holt':
    proj = HoltProjection(params=model_cfg.get('params', {}), train_mode=train_mode,
                          train_fraction=train_fraction, train_end=train_end)
else:
    raise ValueError("MODELO inválido. Use 'prophet', 'arima' ou 'holt'.")

# Loop pelos horizontes definidos no horizons.json
for key, params in horizons_cfg.items():
    # selecionar parâmetros adequados conforme modelo
    if MODEL == 'prophet':
        p = params.get('prophet', {})
        periods = p.get('periods')
        freq = p.get('freq')
    else:
        p = params.get('other', {})
        periods = p.get('periods')
        freq = p.get('freq')

    out_png = f"forecast_{MODEL}_{key}.png"

    # gerar forecast via classe
    forecast_df = proj.forecast(df, periods=periods, freq=freq)

    # preparar df para plot: prophet usa timestamps originais; outros reamostram
    if MODEL == 'prophet':
        df_obs_for_plot = df.copy()
    else:
        df_rs = df.set_index('ds').resample(freq).mean()
        df_rs['y'] = df_rs['y'].interpolate(method='time').ffill().bfill()
        df_rs = df_rs.reset_index()
        df_obs_for_plot = df_rs

    # garantir colunas e ordenação
    forecast_df = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].sort_values('ds').reset_index(drop=True)

    # plot
    plot_standard(df_obs_for_plot, forecast_df, key, MODEL, out_png)

    # imprimir última parte do forecast
    tail_n = periods + 3 if periods is not None else 5
    print(f"\nAmostra do forecast ({MODEL}, {key}) — últimas linhas:")
    print(forecast_df[['ds','yhat','yhat_lower','yhat_upper']].tail(tail_n).to_string(index=False))