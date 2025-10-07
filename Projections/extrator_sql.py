#!/usr/bin/env python3
"""
extrator_sql.py - Extrator minimalista que aceita params literais ou fragmentos SQL (raw).

Config esperado:
 - extractor.query.template_lines contém placeholders nomeados: {equipment_code}, {variable_id}, {date_from}
 - extractor.query.params tem valores simples ou {"raw": "<SQL_FRAGMENT>"}
 - extractor.query.param_order define a ordem lógica dos parâmetros (para binding)

Comportamento:
 - Substitui placeholders por "?" para parâmetros que serão bindados, ou pelo fragmento SQL para params raw.
 - Executa cur.execute(query, tuple(bind_values))
 - Grava CSV.gz e um meta JSON ao lado.

Requisitos:
  pip install pyodbc
"""

import json
import os
import csv
import gzip
import pyodbc
from datetime import datetime, timezone

CONFIG_PATH = "config.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def param_values_from_config(qcfg):
    order = qcfg.get("param_order", [])
    params = qcfg.get("params", {})
    # retorna (mapping_for_format, bind_values_tuple)
    fmt_map = {}
    bind_values = []
    for name in order:
        val = params.get(name)
        if isinstance(val, dict) and "raw" in val:
            # raw SQL fragment: inject inline (no binding)
            fmt_map[name] = val["raw"]
        else:
            # use binding placeholder and append to bind list
            fmt_map[name] = "?"
            bind_values.append(val)
    return fmt_map, tuple(bind_values)

def format_value_for_csv(v):
    if v is None:
        return ""
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    return v

def resolve_credentials(conn_cfg):
    env_user_var = conn_cfg.get("env_user_var", "EXTRACTOR_DB_USER")
    env_pass_var = conn_cfg.get("env_pass_var", "EXTRACTOR_DB_PASS")
    user_env = os.getenv(env_user_var)
    pass_env = os.getenv(env_pass_var)
    if user_env and pass_env:
        return user_env, pass_env
    cred_file = conn_cfg.get("credentials_file", "credenciais.json")
    if os.path.exists(cred_file):
        data = load_json(cred_file)
        creds = data.get("extractor_credentials", {})
        u = creds.get("username")
        p = creds.get("password")
        if u and p:
            return u, p
    raise RuntimeError("Credenciais não encontradas (variáveis de ambiente ou arquivo).")

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

    # Monta mapa de formatação e bind_values simples (muito direto)
    fmt_map, bind_values = param_values_from_config(q_cfg)
    # junta as linhas com placeholders nomeados
    query_template = "\n".join(line.rstrip() for line in query_lines)
    # formata: params raw serão substituídos por seu fragmento SQL; os outros por '?'
    query = query_template.format(**fmt_map)

    os.makedirs(output_dir, exist_ok=True)

    conn_str = f"DRIVER={{{driver}}};SERVER={server},{port};DATABASE={database};UID={username};PWD={password}"

    total_written = 0
    columns_info = []
    sample_python_types = {}

    ts_now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    print(f"[{ts_now}] Iniciando extração. Output: {gzip_path}")
    print(f"Query (preview):\n{query}")

    with pyodbc.connect(conn_str, autocommit=True) as conn:
        with conn.cursor() as cur:
            if bind_values:
                cur.execute(query, bind_values)
            else:
                cur.execute(query)
            desc = cur.description
            if not desc:
                with gzip.open(gzip_path, "wt", encoding="utf-8") as _:
                    pass
                meta = {
                    "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "rows_written": 0,
                    "columns": [],
                    "query": query,
                    "params": q_cfg.get("params", {})
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
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rows_written": total_written,
        "columns": columns_info,
        "sample_python_types": sample_python_types,
        "query": query,
        "params": q_cfg.get("params", {}),
        "output_path": gzip_path,
        "server": server,
        "database": database
    }
    with open(meta_path, "w", encoding="utf-8") as fm:
        json.dump(meta, fm, indent=2, ensure_ascii=False)

    ts_now2 = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    print(f"[{ts_now2}] Extração concluída. {total_written} linhas -> {gzip_path}")
    print(f"Metadados: {meta_path}")

if __name__ == "__main__":
    run_extract()