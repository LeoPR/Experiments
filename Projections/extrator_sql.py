#!/usr/bin/env python3
"""
extrator_sql.py - Extrator minimalista em blocos que lê config.json.

- Lê config.json (seção extractor)
- Monta a query a partir de template_lines (lista) e executa com parâmetros (ordenados por param_order)
- Escreve CSV gz em ./dados/<output_basename>.csv.gz
- Gera metadados JSON ao lado do arquivo com colunas, type_code e amostra de tipo Python
Requisitos: pip install pyodbc
"""

import json
import os
import csv
import gzip
import pyodbc
from datetime import datetime

CONFIG_PATH = "config.json"

def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_query_from_lines(lines):
    # junta as linhas com quebras de linha para ler bem no SQL Server
    return "\n".join(line.rstrip() for line in lines)

def param_values_from_config(qcfg):
    order = qcfg.get("param_order", [])
    params = qcfg.get("params", {})
    return tuple(params.get(k) for k in order)

def format_value_for_csv(v):
    # formatação simples: datas sem micros, None -> ''
    if v is None:
        return ""
    # datetimes em objetos retornados pelo driver têm strftime
    if hasattr(v, "strftime"):
        try:
            return v.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(v)
    return v

def write_metadata(meta_path, meta):
    try:
        with open(meta_path, "w", encoding="utf-8") as fm:
            json.dump(meta, fm, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Aviso: falha ao gravar metadados '{meta_path}': {e}")

def run_extract():
    cfg = load_config(CONFIG_PATH)
    ext = cfg.get("extractor", {})
    conn_cfg = ext.get("connection", {})
    q_cfg = ext.get("query", {})
    out_cfg = ext.get("output", {})
    t_cfg = ext.get("transfer", {})

    server = conn_cfg.get("server")
    database = conn_cfg.get("database")
    username = conn_cfg.get("username")
    password = conn_cfg.get("password")
    driver = conn_cfg.get("driver", "SQL Server")
    port = conn_cfg.get("port", 1433)

    output_dir = out_cfg.get("output_dir", "./dados")
    basename = out_cfg.get("output_basename", "product_extraction")
    gzip_path = os.path.join(output_dir, basename + ".csv.gz")
    meta_path = os.path.join(output_dir, basename + ".csv.gz.meta.json")

    batch_size = int(t_cfg.get("batch_size", 1000))

    query_lines = q_cfg.get("template_lines", [])
    if not query_lines:
        raise ValueError("Config: extractor.query.template_lines vazio.")

    query = build_query_from_lines(query_lines)
    params = param_values_from_config(q_cfg)

    os.makedirs(output_dir, exist_ok=True)

    conn_str = f"DRIVER={{{driver}}};SERVER={server},{port};DATABASE={database};UID={username};PWD={password}"

    total_written = 0
    columns_info = []
    sample_python_types = {}

    print(f"[{datetime.utcnow().isoformat()}Z] Iniciando extração. Output: {gzip_path}")
    print(f"Query executada (primeiras 300 chars):\n{query[:300]}")

    with pyodbc.connect(conn_str, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            desc = cur.description
            if not desc:
                print("Aviso: descrição vazia. Nenhuma coluna retornada.")
                # ainda criamos arquivo vazio com header vazio
                with gzip.open(gzip_path, "wt", encoding="utf-8", newline="") as f:
                    pass
                meta = {
                    "fetched_at": datetime.utcnow().isoformat() + "Z",
                    "rows_written": 0,
                    "columns": [],
                    "query": query,
                    "params": params
                }
                write_metadata(meta_path, meta)
                print("Extração finalizada: 0 linhas.")
                return

            col_names = [c[0] for c in desc]
            # store type_code and info from description
            for c in desc:
                # c structure: (name, type_code, display_size, internal_size, precision, scale, null_ok)
                col_meta = {
                    "name": c[0],
                    "type_code": repr(c[1]) if c[1] is not None else None,
                    "display_size": c[2],
                    "internal_size": c[3],
                    "precision": c[4],
                    "scale": c[5],
                    "null_ok": c[6]
                }
                columns_info.append(col_meta)

            # open gzip and write header + rows in batches
            with gzip.open(gzip_path, "wt", encoding="utf-8", newline="") as gz:
                writer = csv.writer(gz)
                writer.writerow(col_names)

                # fetch in batches
                while True:
                    rows = cur.fetchmany(batch_size)
                    if not rows:
                        break
                    for r in rows:
                        row_out = []
                        for idx, val in enumerate(r):
                            # collect python type sample for metadata (first non-null)
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
        "output_path": gzip_path
    }
    write_metadata(meta_path, meta)
    print(f"[{datetime.utcnow().isoformat()}Z] Extração concluída. {total_written} linhas -> {gzip_path}")
    print(f"Metadados salvos em: {meta_path}")

if __name__ == "__main__":
    run_extract()