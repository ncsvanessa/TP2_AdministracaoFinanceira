"""Estatísticas descritivas (Cap. 10), beta por regressão (Cap. 11.3) e CAPM (Cap. 11.4).

Convenção de anualização (aritmética, a mesma do livro para médias e
volatilidades históricas): média anual = 12 × média mensal; volatilidade
anual = √12 × desvio-padrão mensal (supõe retornos mensais independentes).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as cfg

log = logging.getLogger(__name__)
M = cfg.MESES_POR_ANO


@dataclass
class CenarioCAPM:
    nome: str
    premio_aa: float          # E[R_m] - r_f, anual
    retorno_capm_aa: pd.Series  # E[R_i] = r_f + β_i × prêmio, anual


@dataclass
class ResultadoMetricas:
    descritivas: pd.DataFrame
    cov_m: pd.DataFrame           # covariância mensal
    corr: pd.DataFrame
    regressoes: pd.DataFrame
    excessos: pd.DataFrame        # R_i - r_f (para os gráficos de regressão)
    excesso_mercado: pd.Series    # R_m - r_f
    rf_aa: float
    premio_hist_aa: float
    premio_implausivel: bool
    cenarios: list[CenarioCAPM]

    @property
    def cenario_otimizacao(self) -> CenarioCAPM:
        """Cenário CAPM usado como vetor de retornos na otimização.

        Se o prêmio histórico for implausível, usa o de referência: com prêmio
        ≤ 0 todo ativo com β > 0 teria E[R] ≤ r_f e a carteira tangente não
        existiria.
        """
        return self.cenarios[-1]


def descritivas(retornos: pd.DataFrame, mercado: pd.Series) -> pd.DataFrame:
    """Média e desvio-padrão amostral (ddof=1), mensais e anualizados."""
    dados = retornos.join(mercado.rename(cfg.BENCHMARK))
    media, dp = dados.mean(), dados.std(ddof=1)
    return pd.DataFrame({
        "media_m": media, "vol_m": dp,
        "media_aa": media * M, "vol_aa": dp * np.sqrt(M),
    })


def regressao_beta(retornos: pd.DataFrame, mercado: pd.Series, rf: pd.Series
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """OLS de (R_i - r_f) contra (R_m - r_f) para cada ativo.

    O intercepto é o alfa de Jensen (mensal). Erros-padrão OLS clássicos.
    """
    exc = retornos.sub(rf, axis=0)
    exc_m = (mercado - rf).rename("R_m - r_f")
    X = sm.add_constant(exc_m)
    alpha_ic = 1 - cfg.NIVEL_CONFIANCA

    linhas = {}
    for t in exc.columns:
        fit = sm.OLS(exc[t], X).fit()
        ic = fit.conf_int(alpha=alpha_ic).loc["R_m - r_f"]
        linhas[t] = {
            "beta": fit.params["R_m - r_f"],
            "ep_beta": fit.bse["R_m - r_f"],
            "beta_ic_inf": ic.iloc[0], "beta_ic_sup": ic.iloc[1],
            "alfa_m": fit.params["const"], "alfa_aa": fit.params["const"] * M,
            "p_alfa": fit.pvalues["const"],
            "r2": fit.rsquared, "n": int(fit.nobs),
        }
    return pd.DataFrame(linhas).T.astype(float), exc, exc_m


def retorno_capm(betas: pd.Series, rf_aa: float, premio_aa: float) -> pd.Series:
    """SML (Cap. 11.4): E[R_i] = r_f + β_i (E[R_m] - r_f)."""
    return rf_aa + betas * premio_aa


def calcular(retornos: pd.DataFrame, mercado: pd.Series, rf: pd.Series) -> ResultadoMetricas:
    desc = descritivas(retornos, mercado)
    cov_m = retornos.cov(ddof=1)
    corr = retornos.corr()
    reg, exc, exc_m = regressao_beta(retornos, mercado, rf)

    rf_aa = rf.mean() * M
    premio_hist = exc_m.mean() * M
    implausivel = premio_hist < cfg.PREMIO_MINIMO_PLAUSIVEL_AA

    cenarios = [CenarioCAPM("Prêmio histórico", premio_hist,
                            retorno_capm(reg["beta"], rf_aa, premio_hist))]
    if implausivel:
        log.warning("Prêmio de mercado histórico = %.2f%% a.a. (< %.0f%%): rodando também "
                    "cenário de referência de %.1f%% a.a.", premio_hist * 100,
                    cfg.PREMIO_MINIMO_PLAUSIVEL_AA * 100, cfg.PREMIO_REFERENCIA_AA * 100)
        cenarios.append(CenarioCAPM("Prêmio de referência", cfg.PREMIO_REFERENCIA_AA,
                                    retorno_capm(reg["beta"], rf_aa, cfg.PREMIO_REFERENCIA_AA)))

    # Checagem de consistência: como a reta OLS passa pelas médias, no cenário
    # histórico (média de R_i) - (retorno CAPM) deve ser exatamente 12 × α.
    dif = desc.loc[retornos.columns, "media_aa"] - cenarios[0].retorno_capm_aa
    if not np.allclose(dif, reg["alfa_aa"], atol=1e-10):
        raise AssertionError("Inconsistência entre alfa da regressão e distância à SML.")

    return ResultadoMetricas(desc, cov_m, corr, reg, exc, exc_m, rf_aa, premio_hist,
                             implausivel, cenarios)
