#!/usr/bin/env python3
"""
infer_minimal.py - Wrapper ultra-minimal (mostra APENAS previsões futuras).

Exemplos:
- start como string ISO:
    MinimalInfer().run(equipment="Medidor de Energia P9-QDFI",
                       start="2025-10-07T12:00:00",
                       horizon="1H", resolution="15min")
- start como agora:
    MinimalInfer().run(equipment="Medidor de Energia P9-QDFI",
                       start=None,
                       horizon="1H", resolution="15min")
"""
import json
import pandas as pd
from datetime import datetime
from model import load_model, project

class MinimalInfer:
    def run(self, equipment=None, start=None, horizon="1H", resolution="15min"):
        model, model_path, last_ds = load_model(equipment)
        start_ts = pd.to_datetime(start) if start is not None else (last_ds if last_ds is not None else pd.Timestamp.now())
        forecast_df, _, meta = project(model, start_ts, horizon, resolution, obs_df=None)
        # formatar ds como ISO e imprimir apenas os pontos futuros (project com obs_df=None já retorna só fut_pred)
        rows = []
        for r in forecast_df.to_dict(orient="records"):
            r["ds"] = pd.to_datetime(r["ds"]).isoformat()
            rows.append(r)
        out = {"model": model_path, "equipment": equipment, "start_used": pd.to_datetime(start_ts).isoformat(),
               "horizon": horizon, "resolution": resolution, "forecast": rows}
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return out

if __name__ == "__main__":
    # Exemplo com start atual
    now_iso = datetime.now().isoformat(timespec="seconds")
    MinimalInfer().run(equipment="Medidor de Energia P9-QDFI", start=now_iso, horizon="1H", resolution="15min")