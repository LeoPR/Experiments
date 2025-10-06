#!/usr/bin/env python3
"""
extrator_sql.py - Extrator minimalista em blocos que lê config.json e credenciais externas.

Mudanças:
- Removeu user/password do config principal.
- Lê credenciais de:
   1) Variáveis de ambiente (EXTRACTOR_DB_USER / EXTRACTOR_DB_PASS) se setadas
   2) Arquivo credenciais.json (chave extractor_credentials.username/password)
   3) Erro se não encontrar

Requisitos:
  pip install pyodbc
Uso:
  python extrator_sql.py
"""

import json
import os
import csv
import gzip
import pyodbc
from datetime import datetime

CONFIG_PATH = "config.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def param_values_from_config(qcfg):
    order = qcfg.get("param_order", [])
    params = qcfg.get("params", {})
    return tuple(params.get(k) for k in order)

def format_value_for_csv(v):
    if v is None:
        return ""
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    return v

def resolve_credentials(conn_cfg):
    """
    Ordem:
      1. Variáveis de ambiente (env_user_var/env_pass_var)
      2. Arquivo credenciais.json (credentials_file)
    """
    env_user_var = conn_cfg.get("env_user_var", "EXTRACTOR_DB_USER")
    env_pass_var = conn_cfg.get("env_pass_var", "EXTRACTOR_DB_PASS")

    user_env = os.getenv(env_user_var)
    pass_env = os.getenv(env_pass_var)

    if user_env and pass_env:
        return user_env, pass_env

    cred_file = conn_cfg.get("credentials_file", "credenciais.json")
    if os.path.exists(cred_file):
        try:
            data = load_json(cred_file)
            creds = data.get("extractor_credentials", {})
            u = creds.get("username")
            p = creds.get("password")
            if u and p:
                return u, p
        except Exception as e:
            raise RuntimeError(f"Falha ao ler {cred_file}: {e}")

    raise RuntimeError("Credenciais não encontradas (nem variáveis de ambiente nem arquivo de credenciais).")

def run_extract():
    cfg = load_json(CONFIG_PATH)
    ext = cfg.get("extractor", {})
    conn_cfg = ext.get("connection", {})
    q_cfg = ext.get("query", {})
    out_cfg = ext.get("output", {})
    t_cfg = ext.get("transfer", {})

    server = conn_cfg.get("server")
    database = conn_cfg.get("database")
    driver = conn_cfg.get("driver", "SQL Server")
    port = conn_cfg.get("port", 1433)

    username, password = resolve_credentials(conn_cfg)

    output_dir = out_cfg.get("output_dir", "./dados")
    basename = out_cfg.get("output_basename", "product_extraction")
    gzip_path = os.path.join(output_dir, basename + ".csv.gz")
    meta_path = os.path.join(output_dir, basename + ".csv.gz.meta.json")

    batch_size = int(t_cfg.get("batch_size", 1000))

    query_lines = q_cfg.get("template_lines", [])
    if not query_lines:
        raise ValueError("Config: extractor.query.template_lines vazio.")
    query = "\n".join(line.rstrip() for line in query_lines)
    params = param_values_from_config(q_cfg)

    os.makedirs(output_dir, exist_ok=True)

    conn_str = (
        f"DRIVER={{{driver}}};SERVER={server},{port};DATABASE={database};"
        f"UID={username};PWD={password}"
    )

    total_written = 0
    columns_info = []
    sample_python_types = {}

    print(f"[{datetime.utcnow().isoformat()}Z] Iniciando extração. Output: {gzip_path}")
    print(f"Query (primeiras 200 chars):\n{query[:200]}")

    with pyodbc.connect(conn_str, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            desc = cur.description
            if not desc:
                with gzip.open(gzip_path, "wt", encoding="utf-8", newline="") as _:
                    pass
                meta = {
                    "fetched_at": datetime.utcnow().isoformat() + "Z",
                    "rows_written": 0,
                    "columns": [],
                    "query": query,
                    "params": params
                }
                with open(meta_path, "w", encoding="utf-8") as fm:
                    json.dump(meta, fm, indent=2, ensure_ascii=False)
                print("Extração finalizada: 0 linhas.")
                return

            col_names = [c[0] for c in desc]
            for c in desc:
                columns_info.append({
                    "name": c[0],
                    "type_code": repr(c[1]) if c[1] is not None else None,
                    "display_size": c[2],
                    "internal_size": c[3],
                    "precision": c[4],
                    "scale": c[5],
                    "null_ok": c[6]
                })

            with gzip.open(gzip_path, "wt", encoding="utf-8", newline="") as gz:
                writer = csv.writer(gz)
                writer.writerow(col_names)

                while True:
                    rows = cur.fetchmany(batch_size)
                    if not rows:
                        break
                    for r in rows:
                        row_out = []
                        for idx, val in enumerate(r):
                            if col_names[idx] not in sample_python_types and val is not None:
                                sample_python_types[col_names[idx]] = type(val).__name__
                            row_out.append(format_value_for_csv(val))
                        writer.writerow(row_out)
                        total_written += 1

    meta = {
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "rows_written": total_written,
        "columns": columns_info,
        "sample_python_types": sample_python_types,
        "query": query,
        "params": params,
        "output_path": gzip_path,
        "server": server,
        "database": database
    }
    with open(meta_path, "w", encoding="utf-8") as fm:
        json.dump(meta, fm, indent=2, ensure_ascii=False)

    print(f"[{datetime.utcnow().isoformat()}Z] Extração concluída. {total_written} linhas -> {gzip_path}")
    print(f"Metadados: {meta_path}")

if __name__ == "__main__":
    run_extract()