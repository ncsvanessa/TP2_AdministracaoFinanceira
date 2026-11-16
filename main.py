"""Otimizador de carteira com ações da B3 (Markowitz + CAPM).

Uso:
    python main.py               # usa o cache em data/ se bater com config.py; senão baixa
    python main.py --atualizar   # força novo download
    python main.py --sintetico   # testa o pipeline com dados simulados (sem rede)
"""
from __future__ import annotations

import argparse
import logging
import sys

import config as cfg
import data
import metrics
import optimize
import report

log = logging.getLogger("main")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atualizar", action="store_true", help="baixa os dados de novo, ignorando o cache")
    ap.add_argument("--sintetico", action="store_true", help="usa dados simulados (teste sem rede)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    # 1. Dados
    dados = data.gerar_sinteticos() if args.sintetico else data.obter_dados(args.atualizar)

    # 2–4. Estatísticas, beta e CAPM
    met = metrics.calcular(dados.retornos, dados.mercado, dados.rf)
    log.info("r_f médio = %.2f%% a.a.; prêmio histórico = %.2f%% a.a.", met.rf_aa * 100, met.premio_hist_aa * 100)

    # 5. Markowitz com μ histórico e com μ do CAPM (mesma covariância)
    cov_aa = met.cov_m * cfg.MESES_POR_ANO
    betas = met.regressoes["beta"]
    mu_hist = met.descritivas.loc[dados.retornos.columns, "media_aa"]
    cen = met.cenario_otimizacao
    resultados = [
        optimize.otimizar("Histórico", mu_hist, cov_aa, met.rf_aa, betas),
        optimize.otimizar(f"CAPM ({cen.nome.lower()}, {cen.premio_aa:.1%} a.a.)",
                          cen.retorno_capm_aa, cov_aa, met.rf_aa, betas),
    ]
    for r in resultados:
        tg = r.tangente
        log.info("[%s] tangente: E[R]=%.2f%% σ=%.2f%% Sharpe=%.3f β=%.2f", r.cenario,
                 tg.retorno * 100, tg.vol * 100, tg.sharpe, tg.beta)

    # 6. Relatório
    destino = cfg.RAIZ / "relatorio_sintetico.html" if args.sintetico else cfg.ARQ_RELATORIO
    report.gerar(dados, met, resultados, destino)
    log.info("Relatório gerado: %s", destino)
    return 0


if __name__ == "__main__":
    sys.exit(main())
