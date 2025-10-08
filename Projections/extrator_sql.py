#!/usr/bin/env python3
"""
extrator_sql.py - Extrator simplificado com detecção de parâmetros robusta.

Alteração principal (aprovada):
- param_values_from_config agora recebe também a query_template e:
  * ignora placeholders que aparecem apenas em comentários de linha ("-- ...")
  * usa extractor.query.param_order se presente, senão usa a ordem das chaves em extractor.query.params
  * só coloca "?" (e adiciona bind value) para parâmetros efetivamente presentes na SQL ativa
  * injeta fragments "raw" inline quando configurado

Comportamento:
- Monta a query a partir de template_lines e params (suporta params raw vs bind).
- Executa via pyodbc (bind apenas para os parâmetros ativos).
- Grava CSV.gz com as colunas retornadas pelo SELECT (não faz pós-filtragem).
- Gera meta JSON com informações básicas.

Mudanças intencionais: mínima alteração no fluxo original — substituí apenas a forma como os placeholders e bind values são determinados, para evitar o erro ao comentar parâmetros na SQL.

Uso:
  pip install pyodbc
  python extrator_sql.py
"""

import os
import json
import csv
import gzip
import pyodbc
from datetime import datetime, timezone

CONFIG_PATH = "config.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def param_values_from_config(qcfg, query_template):
    """
    Retorna (fmt_map, bind_values_tuple).

    Regras:
    - order = qcfg.param_order se presente, senão usa a ordem natural de qcfg.params (inserção no JSON).
    - antes de decidir se um placeholder está ativo, removemos os comentários de linha (parte após "--")
      e só então procuramos por ocorrências de {name}.
    - se placeholder não aparece na query (sem comentários) então substituímos por "" e NÃO adicionamos bind.
    - se o valor é {"raw": "..."} então substituímos inline pelo fragmento raw (sem bind).
    - caso contrário substituímos por "?" e adicionamos o valor à lista de bind_values na ordem.
    """
    params = qcfg.get("params", {}) or {}
    # determinar ordem: param_order explicito ou ordem das chaves de params
    order = qcfg.get("param_order")
    if order is None:
        # preserva a ordem de inserção do dict (Python 3.7+ mantém ordem)
        order = list(params.keys())

    # criar versão da query sem comentários de linha para detectar placeholders ativos
    # removemos tudo após '--' em cada linha (simples e suficiente para comentários SQL de linha)
    template_no_comments = "\n".join(line.split("--", 1)[0] for line in query_template.splitlines())

    fmt_map = {}
    bind_values = []
    for name in order:
        val = params.get(name)
        placeholder = "{" + name + "}"
        # se placeholder não aparece na SQL ativa, não bindamos nem deixamos '?'
        if placeholder not in template_no_comments:
            fmt_map[name] = ""  # remove o placeholder se ainda existir no template
            continue

        # placeholder está ativo: decidir raw vs bind
        if isinstance(val, dict) and "raw" in val:
            fmt_map[name] = val["raw"]
        else:
            fmt_map[name] = "?"
            bind_values.append(val)
    return fmt_map, tuple(bind_values)

def format_value_for_csv(v):
    if v is None:
        return ""
    # datetimes -> formato legível (sem timezone)
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    return v

def resolve_credentials(conn_cfg):
    """
    Resolve credenciais por variáveis de ambiente ou por arquivo credenciais.json (configured).
    Lança RuntimeError se não encontrar.
    """
    env_user_var = conn_cfg.get("env_user_var", "EXTRACTOR_DB_USER")
    env_pass_var = conn_cfg.get("env_pass_var", "EXTRACTOR_DB_PASS")
    u = os.getenv(env_user_var)
    p = os.getenv(env_pass_var)
    if u and p:
        return u, p
    cred_file = conn_cfg.get("credentials_file", "credenciais.json")
    if os.path.exists(cred_file):
        data = load_json(cred_file)
        creds = data.get("extractor_credentials", {}) or {}
        cu = creds.get("username")
        cp = creds.get("password")
        if cu and cp:
            return cu, cp
    raise RuntimeError("Credenciais não encontradas (variáveis de ambiente ou arquivo).")

def run_extract():
    cfg = load_json(CONFIG_PATH)
    ext = cfg.get("extractor", {}) or {}
    conn_cfg = ext.get("connection", {}) or {}
    q_cfg = ext.get("query", {}) or {}
    out_cfg = ext.get("output", {}) or {}
    t_cfg = ext.get("transfer", {}) or {}

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

    # juntar template e determinar fmt_map/bind_values com a nova rotina
    query_template = "\n".join(line.rstrip() for line in query_lines)
    fmt_map, bind_values = param_values_from_config(q_cfg, query_template)
    query = query_template.format(**fmt_map)

    os.makedirs(output_dir, exist_ok=True)

    conn_str = f"DRIVER={{{driver}}};SERVER={server},{port};DATABASE={database};UID={username};PWD={password};TrustServerCertificate=yes"

    total_written = 0
    columns_info = []
    sample_python_types = {}

    now_z = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    print(f"[{now_z}] Iniciando extração. Output: {gzip_path}")
    print("Query (preview):")
    print(query)

    with pyodbc.connect(conn_str, autocommit=True) as conn:
        with conn.cursor() as cur:
            # executar com bind se houver
            if bind_values:
                cur.execute(query, bind_values)
            else:
                cur.execute(query)

            desc = cur.description
            if not desc:
                # sem colunas -> gerar arquivo vazio e meta
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
                print(f"[{datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}] Extração finalizada: 0 linhas.")
                return

            # cabeçalho a partir do cursor.description (mantém ordem do SELECT)
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

            # escrita em streaming
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
                            # registrar tipo de amostra para meta (primeira ocorrência)
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

    print(f"[{datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}] Extração concluída. {total_written} linhas -> {gzip_path}")
    print(f"Metadados: {meta_path}")

if __name__ == "__main__":
    run_extract()