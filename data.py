"""Coleta de preços (yfinance) e da Selic (API SGS/BCB), com cache em CSV."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import requests

import config as cfg

log = logging.getLogger(__name__)


@dataclass
class DadosMercado:
    """Retornos mensais alinhados, prontos para a análise."""

    retornos: pd.DataFrame          # ativos (colunas = tickers), retorno simples mensal
    mercado: pd.Series              # retorno mensal do benchmark
    rf: pd.Series                   # Selic mensal, decimal
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Coleta
# ----------------------------------------------------------------------------
def _fim_de_mes(datas: pd.Index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(datas).to_period("M").to_timestamp(how="end").normalize()


def baixar_precos(tickers: list[str], inicio: str, fim: str) -> tuple[pd.DataFrame, dict]:
    """Baixa preços diários ajustados e devolve o último preço de cada mês.

    Inclui o mês anterior a `inicio` para que o primeiro retorno mensal
    (de `inicio`) possa ser calculado.
    """
    import yfinance as yf  # import local: o modo sintético não precisa de rede

    ini = (pd.Timestamp(inicio) - pd.DateOffset(months=1, days=10)).strftime("%Y-%m-%d")
    fim_excl = (pd.Timestamp(fim) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # `end` é exclusivo

    # auto_adjust=True: o "Close" já vem ajustado por desdobramentos e proventos
    # (é o default desde a 0.2.51, mas passamos explicitamente e registramos).
    bruto = yf.download(
        tickers, start=ini, end=fim_excl, interval="1d",
        auto_adjust=True, progress=False, group_by="column", threads=True,
    )
    if bruto is None or bruto.empty:
        raise RuntimeError("yfinance não retornou dados. Verifique a conexão e os tickers.")

    fechamento = bruto["Close"] if isinstance(bruto.columns, pd.MultiIndex) else bruto[["Close"]]
    if not isinstance(bruto.columns, pd.MultiIndex):
        fechamento.columns = tickers
    faltando = [t for t in tickers if t not in fechamento.columns or fechamento[t].isna().all()]
    if faltando:
        log.warning("Sem dados no yfinance para: %s", faltando)

    mensal = fechamento.resample("ME").last()
    mensal.index = _fim_de_mes(mensal.index)
    primeiro = pd.Timestamp(inicio).to_period("M") - 1
    ultimo = pd.Timestamp(fim).to_period("M")
    mensal = mensal.loc[primeiro.to_timestamp(how="end").normalize():
                        ultimo.to_timestamp(how="end").normalize()]

    meta = {
        "fonte_precos": "Yahoo Finance via yfinance",
        "yfinance_versao": yf.__version__,
        "auto_adjust": True,
        "baixado_em": datetime.now().isoformat(timespec="seconds"),
    }
    return mensal, meta


def baixar_selic(inicio: str, fim: str, codigo: int = cfg.SGS_CODIGO_SELIC_MENSAL) -> pd.Series:
    """Selic acumulada no mês (série SGS em % a.m.) convertida para decimal."""
    url = cfg.SGS_URL.format(
        codigo=codigo,
        inicio=pd.Timestamp(inicio).replace(day=1).strftime("%d/%m/%Y"),
        fim=pd.Timestamp(fim).strftime("%d/%m/%Y"),
    )
    resp = requests.get(url, timeout=cfg.HTTP_TIMEOUT_S)
    resp.raise_for_status()
    dados = resp.json()
    if not dados:
        raise RuntimeError(f"A série SGS {codigo} veio vazia para o período.")

    df = pd.DataFrame(dados)
    datas = pd.to_datetime(df["data"], format="%d/%m/%Y")
    selic = pd.Series(df["valor"].astype(float).to_numpy() / 100.0,  # % -> decimal
                      index=_fim_de_mes(datas), name="rf")

    lo, hi = cfg.SELIC_MENSAL_FAIXA_OK
    if not selic.between(lo, hi).all():
        raise ValueError(
            f"Série SGS {codigo} fora da faixa esperada para Selic mensal "
            f"({lo:.0%}–{hi:.0%} a.m.): min={selic.min():.4f}, max={selic.max():.4f}. "
            "Confira o código da série e a unidade."
        )
    return selic


# ----------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------
def salvar_cache(precos: pd.DataFrame, selic: pd.Series, meta: dict) -> None:
    cfg.DIR_DADOS.mkdir(parents=True, exist_ok=True)
    precos.to_csv(cfg.ARQ_PRECOS, index_label="data")
    selic.to_csv(cfg.ARQ_SELIC, index_label="data")
    cfg.ARQ_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Dados brutos salvos em %s", cfg.DIR_DADOS)


def ler_cache() -> tuple[pd.DataFrame, pd.Series, dict]:
    precos = pd.read_csv(cfg.ARQ_PRECOS, index_col="data", parse_dates=True)
    selic = pd.read_csv(cfg.ARQ_SELIC, index_col="data", parse_dates=True).squeeze("columns")
    meta = json.loads(cfg.ARQ_META.read_text(encoding="utf-8")) if cfg.ARQ_META.exists() else {}
    return precos, selic, meta


def cache_valido() -> bool:
    """O cache só é reaproveitado se cobrir exatamente os tickers e o período atuais."""
    if not (cfg.ARQ_PRECOS.exists() and cfg.ARQ_SELIC.exists() and cfg.ARQ_META.exists()):
        return False
    meta = json.loads(cfg.ARQ_META.read_text(encoding="utf-8"))
    return (meta.get("tickers") == cfg.TICKERS and meta.get("benchmark") == cfg.BENCHMARK
            and meta.get("inicio") == cfg.DATA_INICIO and meta.get("fim") == cfg.DATA_FIM)


# ----------------------------------------------------------------------------
# Preparação
# ----------------------------------------------------------------------------
def montar_retornos(precos: pd.DataFrame, selic: pd.Series, meta: dict) -> DadosMercado:
    """Calcula retornos mensais simples, trata ausências e alinha com a Selic."""
    retornos = precos.pct_change(fill_method=None).iloc[1:]  # 1ª linha só serve de base
    n_esperado = len(retornos)

    obs = retornos.notna().sum()
    for t, n in obs.items():
        log.info("  %-10s %3d/%d meses com retorno", t, n, n_esperado)

    if cfg.BENCHMARK not in retornos or obs[cfg.BENCHMARK] < cfg.COBERTURA_MINIMA * n_esperado:
        raise RuntimeError(f"Benchmark {cfg.BENCHMARK} sem histórico suficiente.")

    ativos = [t for t in cfg.TICKERS if t in retornos.columns]
    descartados = [t for t in ativos if obs[t] < cfg.COBERTURA_MINIMA * n_esperado]
    if descartados:
        log.warning("Descartados por cobertura < %.0f%%: %s", cfg.COBERTURA_MINIMA * 100, descartados)
    ativos = [t for t in ativos if t not in descartados]
    if len(ativos) < 2:
        raise RuntimeError("Menos de 2 ativos com histórico suficiente.")

    # Amostra comum: só meses em que todos os ativos, o benchmark e a Selic existem.
    base = retornos[ativos + [cfg.BENCHMARK]].join(selic.rename("rf"), how="inner").dropna()
    perdidos = n_esperado - len(base)
    if perdidos:
        log.warning("%d mês(es) removido(s) por dado faltante (amostra comum).", perdidos)
    log.info("Amostra final: %d meses (%s a %s), %d ativos.",
             len(base), base.index[0].strftime("%m/%Y"), base.index[-1].strftime("%m/%Y"), len(ativos))

    suspeitos = base[ativos + [cfg.BENCHMARK]].abs().gt(cfg.RETORNO_MENSAL_SUSPEITO)
    for data, linha in suspeitos.iterrows():
        for t in linha[linha].index:
            log.warning("Retorno mensal extremo em %s %s: %.1f%% (desdobramento não ajustado?)",
                        t, data.strftime("%m/%Y"), base.at[data, t] * 100)

    meta = {**meta, "ativos_usados": ativos, "ativos_descartados": descartados,
            "n_meses": len(base), "obs_por_ativo": {k: int(v) for k, v in obs.items()},
            "primeiro_mes": base.index[0].strftime("%Y-%m"), "ultimo_mes": base.index[-1].strftime("%Y-%m")}
    return DadosMercado(base[ativos], base[cfg.BENCHMARK], base["rf"], meta)


def obter_dados(atualizar: bool = False) -> DadosMercado:
    """Usa o cache se ele bater com a configuração; caso contrário, baixa."""
    if cache_valido() and not atualizar:
        log.info("Usando cache em %s (use --atualizar para baixar de novo).", cfg.DIR_DADOS)
        precos, selic, meta = ler_cache()
    else:
        log.info("Baixando preços (%d tickers + benchmark)...", len(cfg.TICKERS))
        precos, meta = baixar_precos(cfg.TICKERS + [cfg.BENCHMARK], cfg.DATA_INICIO, cfg.DATA_FIM)
        log.info("Baixando Selic (SGS %d)...", cfg.SGS_CODIGO_SELIC_MENSAL)
        selic = baixar_selic(cfg.DATA_INICIO, cfg.DATA_FIM)
        meta |= {"fonte_rf": f"BCB/SGS série {cfg.SGS_CODIGO_SELIC_MENSAL} (Selic acumulada no mês, % a.m.)",
                 "tickers": cfg.TICKERS, "benchmark": cfg.BENCHMARK,
                 "inicio": cfg.DATA_INICIO, "fim": cfg.DATA_FIM}
        salvar_cache(precos, selic, meta)
    return montar_retornos(precos, selic, meta)


# ----------------------------------------------------------------------------
# Dados sintéticos (teste sem rede)
# ----------------------------------------------------------------------------
BETAS_SINTETICOS: dict[str, float] = {
    "PETR4.SA": 1.35, "VALE3.SA": 1.10, "ITUB4.SA": 0.95, "BBDC4.SA": 1.00,
    "WEGE3.SA": 0.75, "ABEV3.SA": 0.55, "B3SA3.SA": 1.25, "RENT3.SA": 1.45,
}


def gerar_sinteticos(n_meses: int = 60, semente: int = cfg.SEMENTE) -> DadosMercado:
    """Modelo de um fator com betas conhecidos: R_i - r_f = a_i + b_i (R_m - r_f) + e_i.

    Serve para testar o pipeline inteiro: a regressão deve recuperar os betas
    de BETAS_SINTETICOS dentro do IC 95% na maioria dos casos.
    """
    rng = np.random.default_rng(semente)
    base = pd.Timestamp(cfg.DATA_INICIO).to_period("M") - 1  # mês-base do 1º retorno
    idx = _fim_de_mes(pd.period_range(base, periods=n_meses + 1, freq="M").to_timestamp())
    rf = pd.Series(rng.uniform(0.007, 0.011, n_meses + 1), index=idx)
    exc_m = rng.normal(0.005, 0.06, n_meses + 1)
    alfas = rng.normal(0.0, 0.003, len(BETAS_SINTETICOS))
    sigma_e = rng.uniform(0.04, 0.08, len(BETAS_SINTETICOS))
    exc = {t: a + b * exc_m + rng.normal(0, s, n_meses + 1)
           for (t, b), a, s in zip(BETAS_SINTETICOS.items(), alfas, sigma_e)}
    ret = pd.DataFrame(exc, index=idx).add(rf, axis=0)
    ret[cfg.BENCHMARK] = exc_m + rf.to_numpy()
    ret.iloc[0] = 0.0  # linha do mês-base: preço inicial = 100
    precos = 100 * (1 + ret).cumprod()

    meta = {"fonte_precos": "SINTÉTICO (modelo de um fator)", "fonte_rf": "SINTÉTICO",
            "betas_verdadeiros": BETAS_SINTETICOS, "sintetico": True}
    return montar_retornos(precos, rf, meta)
