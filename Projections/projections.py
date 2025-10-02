# projections.py
# Classes de projeção: Projection (base), ProphetProjection, ARIMAProjection, HoltProjection
# Agora com métodos save(path) e load(path) usando joblib para persistência simples.

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
import os


class Projection(ABC):
    def __init__(self, params=None, train_mode='all', train_fraction=0.8, train_end=None):
        """
        params: dict com parâmetros específicos do modelo
        train_mode: 'all' | 'fraction' | 'date'
        """
        self.params = params or {}
        self.train_mode = train_mode
        self.train_fraction = train_fraction
        self.train_end = pd.to_datetime(train_end) if train_end is not None else None
        # atributo para armazenar artefato do modelo treinado, se aplicável
        self._fitted = None

    def _get_train_df(self, df_full):
        if self.train_mode == 'all':
            return df_full.copy()
        if self.train_mode == 'fraction':
            n = int(len(df_full) * self.train_fraction)
            if n < 2:
                return df_full.copy()
            return df_full.iloc[:n].copy()
        if self.train_mode == 'date':
            if self.train_end is None:
                return df_full.copy()
            return df_full[df_full['ds'] <= self.train_end].copy()
        return df_full.copy()

    @abstractmethod
    def forecast(self, df_full, periods, freq):
        """
        Recebe df_full com colunas ['ds','y'] e retorna DataFrame com
        ['ds','yhat','yhat_lower','yhat_upper'] ordenado por ds.
        Deve treinar o modelo conforme train_mode antes de prever.
        """
        raise NotImplementedError

    # Persistência mínima usando joblib
    def save(self, path):
        """
        Salva o objeto Projection (incluindo o artefato treinado) em path via joblib.
        path: caminho para arquivo .pkl
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """
        Carrega um objeto salvo via joblib. Retorna a instância carregada.
        """
        return joblib.load(path)


class ProphetProjection(Projection):
    def forecast(self, df_full, periods, freq):
        from prophet import Prophet

        train_df = self._get_train_df(df_full)

        pparams = self.params or {}
        m = Prophet(
            changepoint_prior_scale=pparams.get('changepoint_prior_scale', 0.05),
            seasonality_mode=pparams.get('seasonality_mode', 'additive'),
            yearly_seasonality=pparams.get('yearly_seasonality', True),
            weekly_seasonality=pparams.get('weekly_seasonality', True),
            daily_seasonality=pparams.get('daily_seasonality', False),
            interval_width=pparams.get('interval_width', 0.95)
        )

        # Treina usando train_df
        m.fit(train_df)
        # Guardar artefato interno para possível debug (não usado diretamente)
        self._fitted = m

        # future para o período pedido (a partir do fim do treino)
        future = m.make_future_dataframe(periods=periods, freq=freq)

        forecast_future = m.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        # predição histórica para cobrir datas observadas (para IC histórico)
        df_all_ds = pd.DataFrame({'ds': df_full['ds'].unique()})
        forecast_hist = m.predict(df_all_ds)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        last_train = train_df['ds'].max()
        fhist = forecast_hist[forecast_hist['ds'] <= last_train].copy()
        ffut = forecast_future[forecast_future['ds'] > last_train].copy()
        forecast = pd.concat([fhist, ffut], ignore_index=True).sort_values('ds').reset_index(drop=True)
        return forecast


class ARIMAProjection(Projection):
    def forecast(self, df_full, periods, freq):
        # usa statsmodels.tsa.arima.model.ARIMA
        from statsmodels.tsa.arima.model import ARIMA

        # reamostrar e interpolar mantendo DatetimeIndex
        s_df = df_full.set_index('ds').resample(freq).mean()
        s_df['y'] = s_df['y'].interpolate(method='time').ffill().bfill()
        s = s_df['y']

        # ajustar treino parcial se solicitado
        train_df = self._get_train_df(s.reset_index().rename(columns={'index': 'ds', 0: 'y'}) if isinstance(s, pd.Series) else s_df.reset_index())
        if isinstance(train_df, pd.DataFrame):
            s_train = train_df.set_index('ds')['y']
        else:
            s_train = s

        # parâmetros de busca
        p_max = int(self.params.get('p_max', 2))
        d_max = int(self.params.get('d_max', 1))
        q_max = int(self.params.get('q_max', 2))
        fallback = tuple(self.params.get('fallback', (1, 0, 1)))

        best_aic = np.inf
        best_res = None
        best_order = None

        # busca simples por (p,d,q)
        for p in range(0, p_max + 1):
            for d in range(0, d_max + 1):
                for q in range(0, q_max + 1):
                    try:
                        model = ARIMA(s_train, order=(p, d, q))
                        res = model.fit(method_kwargs={"warn_convergence": False})
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_res = res
                            best_order = (p, d, q)
                    except Exception:
                        continue

        # fallback se nada encontrado
        if best_res is None:
            model = ARIMA(s_train, order=fallback)
            best_res = model.fit(method_kwargs={"warn_convergence": False})
            best_order = fallback

        self._fitted = best_res

        # in-sample
        ins_pred = best_res.get_prediction(start=0, end=len(s_train) - 1)
        ins_mean = ins_pred.predicted_mean
        ins_ci = ins_pred.conf_int(alpha=0.05)

        # out-of-sample
        out_forecast = best_res.get_forecast(steps=periods)
        out_mean = out_forecast.predicted_mean
        out_ci = out_forecast.conf_int(alpha=0.05)

        idx_in = s_train.index
        start_out = idx_in[-1] + pd.tseries.frequencies.to_offset(freq)
        idx_out = pd.date_range(start=start_out, periods=periods, freq=freq) if periods > 0 else pd.DatetimeIndex([])

        df_in = pd.DataFrame({
            'ds': idx_in,
            'yhat': ins_mean.values,
            'yhat_lower': ins_ci.iloc[:, 0].values,
            'yhat_upper': ins_ci.iloc[:, 1].values
        })
        df_out = pd.DataFrame({
            'ds': idx_out,
            'yhat': out_mean.values if periods > 0 else [],
            'yhat_lower': out_ci.iloc[:, 0].values if periods > 0 else [],
            'yhat_upper': out_ci.iloc[:, 1].values if periods > 0 else []
        })
        df_full_forecast = pd.concat([df_in, df_out], ignore_index=True).sort_values('ds').reset_index(drop=True)
        return df_full_forecast


class HoltProjection(Projection):
    def forecast(self, df_full, periods, freq):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        s_df = df_full.set_index('ds').resample(freq).mean()
        s_df['y'] = s_df['y'].interpolate(method='time').ffill().bfill()
        s = s_df['y']

        # aplicar treino parcial se solicitado
        if self.train_mode == 'all':
            s_train = s
        elif self.train_mode == 'fraction':
            n = int(len(s) * self.train_fraction)
            s_train = s.iloc[:n] if n >= 2 else s
        elif self.train_mode == 'date':
            if self.train_end is None:
                s_train = s
            else:
                s_train = s[s.index <= self.train_end] if (s.index <= self.train_end).any() else s
        else:
            s_train = s

        trend = self.params.get('trend', 'add')
        seasonal = self.params.get('seasonal', None)
        init = self.params.get('initialization_method', 'estimated')

        model = ExponentialSmoothing(s_train, trend=trend, seasonal=seasonal, initialization_method=init)
        fit = model.fit()
        fitted = fit.fittedvalues
        out = fit.forecast(periods)

        self._fitted = fit

        idx_in = s_train.index
        start_out = idx_in[-1] + pd.tseries.frequencies.to_offset(freq)
        idx_out = pd.date_range(start=start_out, periods=periods, freq=freq) if periods > 0 else pd.DatetimeIndex([])

        resid = s_train.values - fitted.values
        sigma = np.std(resid) if len(resid) > 1 else 0.0
        z = 1.96

        df_in = pd.DataFrame({
            'ds': idx_in,
            'yhat': fitted.values,
            'yhat_lower': fitted.values - z * sigma,
            'yhat_upper': fitted.values + z * sigma
        })

        df_out = pd.DataFrame({
            'ds': idx_out,
            'yhat': out.values if periods > 0 else [],
            'yhat_lower': out.values - z * sigma if periods > 0 else [],
            'yhat_upper': out.values + z * sigma if periods > 0 else []
        })

        df_full_forecast = pd.concat([df_in, df_out], ignore_index=True).sort_values('ds').reset_index(drop=True)
        return df_full_forecast