"""Parâmetros do estudo. Todos os valores configuráveis ficam aqui."""
from __future__ import annotations

from pathlib import Path

# --- Universo e período -------------------------------------------------------
TICKERS: list[str] = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA",
    "WEGE3.SA", "ABEV3.SA", "B3SA3.SA", "RENT3.SA",
]
BENCHMARK: str = "^BVSP"
DATA_INICIO: str = "2021-11-01"   # primeiro mês com retorno calculado
DATA_FIM: str = "2026-10-31"      # último mês com retorno calculado (60 meses)

# --- Restrições da carteira ---------------------------------------------------
PESO_MIN: float = 0.0             # long-only
PESO_MAX: float | None = 0.30     # teto por ativo; None = sem teto

# --- CAPM ---------------------------------------------------------------------
# Se o prêmio histórico (média de R_m - r_f, anualizada) ficar abaixo deste
# limiar, ele é sinalizado como implausível e roda-se também o cenário de
# referência abaixo.
PREMIO_MINIMO_PLAUSIVEL_AA: float = 0.02
PREMIO_REFERENCIA_AA: float = 0.05

# --- Taxa livre de risco (API SGS do BCB) -----------------------------------
# 4390 = "Taxa de juros - Selic acumulada no mês" (% a.m.).
SGS_CODIGO_SELIC_MENSAL: int = 4390
SGS_URL: str = (
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
    "?formato=json&dataInicial={inicio}&dataFinal={fim}"
)
# Faixa de sanidade da Selic mensal em decimal: se a série vier fora dela,
# provavelmente o código ou a unidade estão errados.
SELIC_MENSAL_FAIXA_OK: tuple[float, float] = (0.0, 0.03)

# --- Tratamento de dados ------------------------------------------------------
COBERTURA_MINIMA: float = 0.90     # fração mínima de meses com preço por ativo
RETORNO_MENSAL_SUSPEITO: float = 0.50  # |r| acima disso vai para o log
HTTP_TIMEOUT_S: int = 30

# --- Otimização ---------------------------------------------------------------
N_PONTOS_FRONTEIRA: int = 50
N_CARTEIRAS_ALEATORIAS: int = 5_000
SEMENTE: int = 42
TOL_SOMA_PESOS: float = 1e-6
TOL_AUTOVALOR: float = -1e-10      # menor autovalor aceito na matriz de cov.
SOLVER_MAXITER: int = 1_000

# --- Convenções ---------------------------------------------------------------
MESES_POR_ANO: int = 12
NIVEL_CONFIANCA: float = 0.95

# --- Caminhos -----------------------------------------------------------------
RAIZ: Path = Path(__file__).resolve().parent
DIR_DADOS: Path = RAIZ / "data"
ARQ_PRECOS: Path = DIR_DADOS / "precos_mensais.csv"
ARQ_SELIC: Path = DIR_DADOS / "selic_mensal.csv"
ARQ_META: Path = DIR_DADOS / "metadados.json"
ARQ_RELATORIO: Path = RAIZ / "relatorio.html"
