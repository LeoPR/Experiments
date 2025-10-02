# forecast.py
# Script principal que usa classes em projections.py, gera 3 PNGs e faz persistência de modelos.
# Agora salva imagens em pasta configurável e respeita o flag save_images no model_store.json.
#
# Uso: coloque projections.py, config.json, horizons.json e model_store.json no mesmo diretório;
# ajuste INPUT_CSV e MODEL conforme necessário; rode no Spyder.

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from projections import ProphetProjection, ARIMAProjection, HoltProjection

# ----- CONFIGURAÇÕES -----
MODEL = 'prophet'   # 'prophet', 'arima' ou 'holt'
INPUT_CSV = './dados/Medidor_Energy_202510011009.csv'
CONFIG_FILE = 'config.json'
HORIZONS_FILE = 'horizons.json'
MODEL_STORE_FILE = 'model_store.json'
# -------------------------

# Carregar config do modelo
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
model_cfg = cfg.get(MODEL, {})

# Carregar horizons
with open(HORIZONS_FILE, 'r', encoding='utf-8') as f:
    horizons_cfg = json.load(f)

# Carregar model store
with open(MODEL_STORE_FILE, 'r', encoding='utf-8') as f:
    store_cfg = json.load(f)

MODEL_DIR = store_cfg.get('global', {}).get('model_dir', './models')
SAVE_IMAGES_FLAG = store_cfg.get('global', {}).get('save_images', True)
IMAGES_DIR = store_cfg.get('global', {}).get('images_dir', './images')
os.makedirs(MODEL_DIR, exist_ok=True)
if SAVE_IMAGES_FLAG:
    os.makedirs(IMAGES_DIR, exist_ok=True)

# Carrega CSV (apenas Date e Value)
df = pd.read_csv(INPUT_CSV, parse_dates=['Date'])
df = df[['Date', 'Value']].rename(columns={'Date': 'ds', 'Value': 'y'})
df = df.sort_values('ds')
df['y'] = pd.to_numeric(df['y'], errors='coerce')


# Função de plot (mesma estética aprovada)
def plot_standard(df_obs, forecast, horizon_label, model_name, out_png, save_images=True, images_dir="./images", show_plot=True):
    last_date = df_obs['ds'].max()
    is_future = forecast['ds'] > last_date
    is_past = ~is_future

    plt.figure(figsize=(11, 6))

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

    if save_images:
        out_path = os.path.join(images_dir, out_png)
        try:
            plt.savefig(out_path)
            print(f"Salvo: {out_path}")
        except Exception as e:
            print(f"Aviso: falha ao salvar imagem {out_path}: {e}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def get_or_train_projection(model_name, proj_class, model_cfg, store_cfg, df_full):
    """
    - model_name: 'prophet'|'arima'|'holt'
    - proj_class: classe de Projection (ProphetProjection, ARIMAProjection, ...)
    - model_cfg: dict do config.json para esse modelo
    - store_cfg: conteúdo de model_store.json
    - df_full: DataFrame completo com ds,y
    Retorna instância treinada (carregada do disco ou treinada e salva).
    """
    model_entry = store_cfg.get('models', {}).get(model_name, {})
    save_flag = model_entry.get('save', True)
    filename = model_entry.get('filename', f"{model_name}_model.pkl")
    model_dir = store_cfg.get('global', {}).get('model_dir', './models')
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, filename)

    # Tentar carregar
    if os.path.exists(path):
        try:
            print(f"Tentando carregar modelo salvo: {path}")
            loaded = proj_class.load(path)
            print("Modelo carregado com sucesso.")
            return loaded
        except Exception as e:
            print(f"Falha ao carregar modelo salvo ({e}), iremos treinar novo modelo.")

    # Se não carregou, instanciar e treinar (treino executado dentro de forecast chamando com periods=0)
    print("Instanciando e treinando modelo...")
    train_mode = model_cfg.get('train_mode', 'all')
    train_fraction = model_cfg.get('train_fraction', 0.8)
    train_end = model_cfg.get('train_end', None)

    proj = proj_class(params=model_cfg.get('params', {}), train_mode=train_mode,
                      train_fraction=train_fraction, train_end=train_end)

    # escolher granularidade de treinamento: prefer default_data_window.read_granularity
    default_gr = store_cfg.get('default_data_window', {}).get('read_granularity', '1min')
    # chamar forecast com periods=0 para forçar treino (sem prever futuros)
    try:
        proj.forecast(df_full, periods=0, freq=default_gr)
    except Exception as e:
        print(f"Erro durante o treino inicial com granularidade {default_gr}: {e}")
        # fallback: usar 'D'
        proj.forecast(df_full, periods=0, freq='D')

    # salvar se configurado
    if save_flag:
        try:
            proj.save(path)
            # grava também um arquivo meta com versão do model_store (se disponível)
            meta_store = {
                "model_name": model_name,
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "model_store_version": store_cfg.get('models', {}).get(model_name, {}).get('version')
            }
            meta_path = path + ".store.meta.json"
            with open(meta_path, "w", encoding="utf-8") as fm:
                json.dump(meta_store, fm, indent=2, ensure_ascii=False)
            print(f"Modelo salvo em: {path}")
        except Exception as e:
            print(f"Falha ao salvar modelo: {e}")

    return proj


# Instanciar/obter projeção treinada
train_mode = model_cfg.get('train_mode', 'all')
train_fraction = model_cfg.get('train_fraction', 0.8)
train_end = model_cfg.get('train_end', None)

if MODEL == 'prophet':
    proj = get_or_train_projection('prophet', ProphetProjection, model_cfg, store_cfg, df)
elif MODEL == 'arima':
    proj = get_or_train_projection('arima', ARIMAProjection, model_cfg, store_cfg, df)
elif MODEL == 'holt':
    proj = get_or_train_projection('holt', HoltProjection, model_cfg, store_cfg, df)
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

    # gerar forecast via classe (proj já treinada)
    forecast_df = proj.forecast(df, periods=periods, freq=freq)

    # Para plot: prophet usa timestamps originais; outros reamostram
    if MODEL == 'prophet':
        df_obs_for_plot = df.copy()
    else:
        df_rs = df.set_index('ds').resample(freq).mean()
        df_rs['y'] = df_rs['y'].interpolate(method='time').ffill().bfill()
        df_rs = df_rs.reset_index()
        df_obs_for_plot = df_rs

    # garantir colunas e ordenação
    forecast_df = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].sort_values('ds').reset_index(drop=True)

    # plot: usa o diretório de imagens configurado no model_store.json
    image_out_name = out_png
    plot_standard(df_obs_for_plot, forecast_df, key, MODEL, image_out_name,
                  save_images=SAVE_IMAGES_FLAG, images_dir=IMAGES_DIR, show_plot=True)

    tail_n = periods + 3 if periods is not None else 5
    print(f"\nAmostra do forecast ({MODEL}, {key}) — últimas linhas:")
    print(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(tail_n).to_string(index=False))