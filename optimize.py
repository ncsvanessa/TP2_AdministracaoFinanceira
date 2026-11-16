"""Otimização de média-variância (Markowitz) com restrições long-only e teto por ativo.

Solver: SLSQP (scipy). Ele aceita ao mesmo tempo limites por variável
(0 ≤ wᵢ ≤ teto) e restrições de igualdade (Σw = 1, w·μ = alvo), e o problema
de mínima variância é um QP convexo pequeno, então converge rápido e para
o ótimo global. A carteira tangente (máximo Sharpe) não é convexa em w, mas
parte do melhor ponto da fronteira, o que evita ótimos locais ruins.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

import config as cfg

log = logging.getLogger(__name__)


@dataclass
class Carteira:
    nome: str
    pesos: pd.Series
    retorno: float
    vol: float
    sharpe: float
    beta: float


@dataclass
class ResultadoOtimizacao:
    cenario: str                  # "Histórico" ou "CAPM (...)"
    mu: pd.Series                 # retornos esperados usados (anuais)
    fronteira: pd.DataFrame       # colunas: retorno, vol, sharpe
    min_var: Carteira
    tangente: Carteira
    aleatorias: pd.DataFrame      # colunas: retorno, vol, sharpe
    rf: float


# ----------------------------------------------------------------------------
# Validações
# ----------------------------------------------------------------------------
def validar_cov(cov: np.ndarray) -> None:
    if not np.allclose(cov, cov.T):
        raise ValueError("Matriz de covariância não é simétrica.")
    menor = np.linalg.eigvalsh(cov).min()
    if menor < cfg.TOL_AUTOVALOR:
        raise ValueError(f"Matriz de covariância não é semidefinida positiva (menor autovalor = {menor:.3e}).")


def validar_pesos(w: np.ndarray, teto: float) -> None:
    if abs(w.sum() - 1) > cfg.TOL_SOMA_PESOS:
        raise ValueError(f"Soma dos pesos = {w.sum():.8f} ≠ 1.")
    if w.min() < cfg.PESO_MIN - cfg.TOL_SOMA_PESOS or w.max() > teto + cfg.TOL_SOMA_PESOS:
        raise ValueError(f"Pesos fora dos limites [{cfg.PESO_MIN}, {teto}]: {np.round(w, 4)}")


# ----------------------------------------------------------------------------
# Blocos
# ----------------------------------------------------------------------------
def _teto(n: int) -> float:
    teto = 1.0 if cfg.PESO_MAX is None else cfg.PESO_MAX
    if teto * n < 1:
        raise ValueError(f"Inviável: {n} ativos × teto {teto:.0%} < 100%.")
    return teto


def _vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(w @ cov @ w, 0.0)))


def _resolver(obj, w0: np.ndarray, teto: float, extras: list[dict] | None = None) -> np.ndarray:
    restr = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}] + (extras or [])
    res = minimize(obj, w0, method="SLSQP", bounds=[(cfg.PESO_MIN, teto)] * len(w0),
                   constraints=restr, options={"maxiter": cfg.SOLVER_MAXITER, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"SLSQP não convergiu: {res.message}")
    w = np.clip(res.x, cfg.PESO_MIN, teto)
    return w / w.sum()


def minima_variancia(cov: np.ndarray, teto: float) -> np.ndarray:
    n = len(cov)
    return _resolver(lambda w: w @ cov @ w, np.full(n, 1 / n), teto)


def retorno_maximo(mu: np.ndarray, teto: float) -> float:
    """Maior retorno atingível sob as restrições (programa linear)."""
    res = linprog(-mu, A_eq=np.ones((1, len(mu))), b_eq=[1.0],
                  bounds=[(cfg.PESO_MIN, teto)] * len(mu), method="highs")
    if not res.success:
        raise RuntimeError(f"linprog falhou: {res.message}")
    return float(mu @ res.x)


def carteiras_aleatorias(mu: np.ndarray, cov: np.ndarray, rf: float, teto: float,
                         n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Pesos ~ Dirichlet(1) (uniformes no simplex), descartando os que violam o teto."""
    k, aceitos = len(mu), []
    while sum(len(a) for a in aceitos) < n:
        w = rng.dirichlet(np.ones(k), size=n)
        aceitos.append(w[w.max(axis=1) <= teto])
    W = np.vstack(aceitos)[:n]
    ret = W @ mu
    vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    return pd.DataFrame({"retorno": ret, "vol": vol, "sharpe": (ret - rf) / vol})


def _carteira(nome: str, w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float,
              betas: np.ndarray, ativos: pd.Index) -> Carteira:
    ret, vol = float(w @ mu), _vol(w, cov)
    # β da carteira = média ponderada dos βs (Cap. 11.3)
    return Carteira(nome, pd.Series(w, index=ativos), ret, vol, (ret - rf) / vol, float(w @ betas))


# ----------------------------------------------------------------------------
# Rotina principal
# ----------------------------------------------------------------------------
def otimizar(cenario: str, mu: pd.Series, cov_aa: pd.DataFrame, rf_aa: float,
             betas: pd.Series) -> ResultadoOtimizacao:
    ativos = mu.index
    m, S, b = mu.to_numpy(), cov_aa.loc[ativos, ativos].to_numpy(), betas[ativos].to_numpy()
    validar_cov(S)
    teto = _teto(len(m))
    rng = np.random.default_rng(cfg.SEMENTE)

    # 1) Variância mínima global
    w_mv = minima_variancia(S, teto)
    validar_pesos(w_mv, teto)
    aleat = carteiras_aleatorias(m, S, rf_aa, teto, cfg.N_CARTEIRAS_ALEATORIAS, rng)
    sigma_mv, sigmas = _vol(w_mv, S), np.sqrt(np.diag(S))
    if teto >= 1.0:
        # Sem teto, cada ativo isolado é viável, então σ_mv ≤ min σ_i.
        ref, desc = sigmas.min(), "min σ_i"
    else:
        # Com teto, um ativo isolado não é viável; a referência passa a ser a menor
        # σ entre as carteiras viáveis conhecidas (igual-ponderada e aleatórias).
        ref, desc = min(_vol(np.full(len(m), 1 / len(m)), S), aleat["vol"].min()), "menor σ viável amostrada"
        log.info("σ_mv = %.4f; min σ_i = %.4f (não comparável diretamente com teto de %.0f%%)",
                 sigma_mv, sigmas.min(), teto * 100)
    if sigma_mv > ref * (1 + 1e-6):
        raise AssertionError(f"Variância mínima inconsistente: σ_mv={sigma_mv:.4f} > {desc}={ref:.4f}")

    # 2) Fronteira eficiente: min σ² s.a. w·μ = alvo, alvos de μ_mv até μ_max
    alvos = np.linspace(float(w_mv @ m), retorno_maximo(m, teto), cfg.N_PONTOS_FRONTEIRA)
    pontos, w_prev, pesos_fronteira = [], w_mv, []
    for alvo in alvos:
        w = _resolver(lambda w: w @ S @ w, w_prev, teto,
                      [{"type": "eq", "fun": lambda w, a=alvo: w @ m - a}])
        validar_pesos(w, teto)
        pontos.append((float(w @ m), _vol(w, S)))
        pesos_fronteira.append(w)
        w_prev = w
    fronteira = pd.DataFrame(pontos, columns=["retorno", "vol"])
    fronteira["sharpe"] = (fronteira["retorno"] - rf_aa) / fronteira["vol"]

    # 3) Tangente: max (w·μ - r_f)/σ, partindo do melhor ponto da fronteira
    if (m <= rf_aa).all():
        log.warning("[%s] Todos os retornos esperados ≤ r_f: a tangente terá Sharpe negativo.", cenario)
    w0 = pesos_fronteira[int(fronteira["sharpe"].idxmax())]
    w_tg = _resolver(lambda w: -(w @ m - rf_aa) / _vol(w, S), w0, teto)
    validar_pesos(w_tg, teto)

    return ResultadoOtimizacao(
        cenario, mu, fronteira,
        _carteira("Variância mínima", w_mv, m, S, rf_aa, b, ativos),
        _carteira("Tangente (máx. Sharpe)", w_tg, m, S, rf_aa, b, ativos),
        aleat, rf_aa,
    )
