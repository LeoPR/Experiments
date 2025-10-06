#!/usr/bin/env python3
"""
train.py - Treina modelos Prophet a partir do CSV extraído pelo extrator.

- Lê config.json para localizar o arquivo extraído e parâmetros de treino.
- Lê o meta do extrator (product_extraction.csv.gz.meta.json) para validar colunas quando disponível.
- Usa y_column e ds_column do config, e carrega extra_columns (se listadas) para uso como variáveis de agrupamento/metadata.
- Treina modelo "general" (média por timestamp) e, se configurado e houver EquipmentName, treina por equipamento.
- Salva modelos em ./models/<slug>_prophet.pkl e metadados .meta.json.

Uso:
  pip install pandas prophet joblib
  python train.py
"""

import os, json
from datetime import datetime
import joblib
import pandas as pd
from prophet import Prophet

# carregar config
with open("config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

ext = cfg.get("extractor", {})
out = ext.get("output", {})
inp_dir = out.get("output_dir", "./dados")
basename = out.get("output_basename", "product_extraction")
INPUT_PATH = os.path.join(inp_dir, basename + ".csv.gz")
META_PATH = os.path.join(inp_dir, basename + ".csv.gz.meta.json")

train_cfg = ext.get("train", {})
Y_COLUMN = train_cfg.get("y_column", "Value")
DS_COLUMN = train_cfg.get("ds_column", "Date")
EXTRA_COLUMNS = train_cfg.get("extra_columns", []) or []
TRAIN_PER_EQUIPMENT = bool(train_cfg.get("train_per_equipment", True))
EQUIPMENT_FILTER = train_cfg.get("equipment_filter")

MODELS_DIR = cfg.get("models_defaults", {}).get("model_dir", "./models")
PROPHET_PARAMS = cfg.get("prophet", {}).get("params", {})

def slugify(s):
    s = str(s or "")
    out = []
    for ch in s.strip():
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
        else:
            out.append("_")
    return "".join(out) or "model"

def train_and_save(df, label):
    os.makedirs(MODELS_DIR, exist_ok=True)
    slug = slugify(label)
    pkl = os.path.join(MODELS_DIR, f"{slug}_prophet.pkl")
    print(f"[{datetime.utcnow().isoformat()}Z] Treinando '{label}' ({len(df)} pontos)...")
    m = Prophet(**PROPHET_PARAMS)
    m.fit(df)
    joblib.dump(m, pkl)
    meta = {"model_label": label, "saved_at": datetime.utcnow().isoformat()+"Z",
            "rows_used": int(len(df)), "last_ds": df['ds'].max().isoformat() if not df['ds'].isnull().all() else None,
            "prophet_params": PROPHET_PARAMS}
    with open(pkl + ".meta.json", "w", encoding="utf-8") as fm:
        json.dump(meta, fm, indent=2, ensure_ascii=False)
    print(f"[{datetime.utcnow().isoformat()}Z] Salvo: {pkl}")
    return pkl

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Arquivo de entrada não encontrado: {INPUT_PATH}")

meta = {}
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as fm:
        meta = json.load(fm)
else:
    print(f"Aviso: metadados não encontrados em {META_PATH}. Continuarei sem validação antecipada.")

available_cols = []
if meta and "columns" in meta:
    available_cols = [c.get("name") for c in meta.get("columns", []) if c.get("name")]

# validações mínimas
if available_cols:
    if Y_COLUMN not in available_cols:
        raise ValueError(f"Coluna y configurada ('{Y_COLUMN}') não encontrada nos metadados: {available_cols}")
    if DS_COLUMN not in available_cols:
        raise ValueError(f"Coluna ds configurada ('{DS_COLUMN}') não encontrada nos metadados: {available_cols}")

# decidir colunas a carregar: ds, y, extras (se disponíveis)
usecols = [DS_COLUMN, Y_COLUMN]
extras_to_use = []
for col in EXTRA_COLUMNS:
    if not available_cols or col in available_cols:
        extras_to_use.append(col)
usecols += extras_to_use
usecols = list(dict.fromkeys(usecols))  # dedup

print(f"[{datetime.utcnow().isoformat()}Z] Carregando {INPUT_PATH} (colunas={usecols}) ...")
df = pd.read_csv(INPUT_PATH, compression='gzip', usecols=usecols, parse_dates=[DS_COLUMN])
df = df.rename(columns={DS_COLUMN: "ds", Y_COLUMN: "y"})
# manter nomes extras como vierem (ex: EquipmentName)
df = df.dropna(subset=["ds", "y"])
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna(subset=["y"])
df = df.sort_values("ds").reset_index(drop=True)

if EQUIPMENT_FILTER:
    print(f"Filtro de equipamento ativo: {EQUIPMENT_FILTER}")
    if "EquipmentName" not in df.columns:
        raise ValueError("Filtro por equipamento solicitado, mas coluna 'EquipmentName' não está presente nos dados.")
    df = df[df["EquipmentName"] == EQUIPMENT_FILTER]
    if len(df) < 2:
        raise ValueError(f"Após aplicar filtro '{EQUIPMENT_FILTER}' restaram {len(df)} pontos — insuficiente para treinar.")

if len(df) < 2:
    raise ValueError("Dados insuficientes para treinar (menos de 2 pontos).")

# Treinar geral: média por timestamp
general = df.groupby("ds")["y"].mean().reset_index().sort_values("ds").reset_index(drop=True)
general_path = train_and_save(general, "general")

specific_models = {}
if TRAIN_PER_EQUIPMENT and "EquipmentName" in df.columns:
    equipments = df["EquipmentName"].dropna().unique().tolist()
    print(f"Treinando modelos por equipamento (encontrados {len(equipments)})...")
    for eq in equipments:
        sub = df[df["EquipmentName"] == eq][["ds", "y"]].sort_values("ds").reset_index(drop=True)
        if len(sub) < 2:
            print(f" - Pulando '{eq}' (insuficiente: {len(sub)})")
            continue
        p = train_and_save(sub, eq)
        specific_models[eq] = p

print("Treino finalizado.")
print("Modelo geral:", general_path)
print("Modelos específicos treinados:", list(specific_models.keys()))