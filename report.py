"""Relatório HTML autocontido (Plotly embutido, sem dependências externas)."""
from __future__ import annotations

import html
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots

import config as cfg
from data import DadosMercado
from metrics import ResultadoMetricas
from optimize import ResultadoOtimizacao

NIVEL_SIG = 1 - cfg.NIVEL_CONFIANCA
PESO_RELEVANTE = 0.05           # peso mínimo para citar um ativo na interpretação
LAYOUT = dict(template="plotly_white", font=dict(family="Arial, sans-serif", size=12),
              margin=dict(l=60, r=30, t=50, b=50))
COR = {"ativo": "#1f4e79", "mercado": "#c00000", "mv": "#2e7d32", "tg": "#e65100",
       "sml": "#555555", "sml_ref": "#8e24aa"}


# ----------------------------------------------------------------------------
# Utilitários
# ----------------------------------------------------------------------------
def pct(x: float, casas: int = 2) -> str:
    return "—" if pd.isna(x) else f"{x * 100:.{casas}f}%".replace(".", ",")


def num(x: float, casas: int = 3) -> str:
    return "—" if pd.isna(x) else f"{x:.{casas}f}".replace(".", ",")


def nome(t: str) -> str:
    return t.replace(".SA", "")


def lista(itens: list[str]) -> str:
    if not itens:
        return "nenhuma"
    return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1]


def fig_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


def tabela(df: pd.DataFrame) -> str:
    return df.to_html(classes="tab", border=0, escape=True)


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
def fig_correlacao(corr: pd.DataFrame) -> go.Figure:
    rotulos = [nome(c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=rotulos, y=rotulos, zmin=-1, zmax=1, colorscale="RdBu_r",
        text=np.round(corr.values, 2), texttemplate="%{text}", colorbar=dict(title="ρ")))
    fig.update_layout(title="Matriz de correlação dos retornos mensais", height=520,
                      yaxis_autorange="reversed", **LAYOUT)
    return fig


def fig_regressoes(met: ResultadoMetricas) -> go.Figure:
    ativos = list(met.excessos.columns)
    ncol = 4
    nlin = math.ceil(len(ativos) / ncol)
    fig = make_subplots(rows=nlin, cols=ncol, subplot_titles=[
        f"{nome(t)}: β={num(met.regressoes.at[t, 'beta'], 2)}, R²={num(met.regressoes.at[t, 'r2'], 2)}"
        for t in ativos], horizontal_spacing=0.06, vertical_spacing=0.12)
    x = met.excesso_mercado
    xs = np.linspace(x.min(), x.max(), 2)
    for i, t in enumerate(ativos):
        r, c = divmod(i, ncol)
        reg = met.regressoes.loc[t]
        fig.add_trace(go.Scatter(x=x, y=met.excessos[t], mode="markers", showlegend=False,
                                 marker=dict(size=5, color=COR["ativo"], opacity=0.6),
                                 hovertemplate="mercado %{x:.1%}<br>ativo %{y:.1%}<extra></extra>"),
                      row=r + 1, col=c + 1)
        fig.add_trace(go.Scatter(x=xs, y=reg["alfa_m"] + reg["beta"] * xs, mode="lines",
                                 showlegend=False, line=dict(color=COR["mercado"])),
                      row=r + 1, col=c + 1)
    fig.update_xaxes(tickformat=".0%", title_text="R_m − r_f", title_font_size=10)
    fig.update_yaxes(tickformat=".0%", title_text="R_i − r_f", title_font_size=10)
    fig.update_layout(title="Regressão do retorno em excesso de cada ação contra o do mercado (mensal)",
                      height=300 * nlin + 60, **LAYOUT)
    return fig


def fig_sml(met: ResultadoMetricas) -> go.Figure:
    reg = met.regressoes
    hist = met.descritivas.loc[reg.index, "media_aa"]
    xs = np.linspace(min(0, reg["beta"].min()) - 0.1, max(1.5, reg["beta"].max() + 0.2), 2)
    fig = go.Figure()
    estilos = [("solid", COR["sml"]), ("dash", COR["sml_ref"])]
    for cen, (traco, cor) in zip(met.cenarios, estilos):
        fig.add_trace(go.Scatter(x=xs, y=met.rf_aa + xs * cen.premio_aa, mode="lines",
                                 name=f"SML – {cen.nome} ({pct(cen.premio_aa, 1)} a.a.)",
                                 line=dict(color=cor, dash=traco)))
    acima = reg["alfa_aa"] > 0
    for flag, rot, simb in [(True, "Acima da SML (α > 0)", "triangle-up"),
                            (False, "Abaixo da SML (α < 0)", "triangle-down")]:
        sel = reg.index[acima == flag]
        fig.add_trace(go.Scatter(
            x=reg.loc[sel, "beta"], y=hist[sel], mode="markers+text", name=rot,
            text=[nome(t) + ("*" if reg.at[t, "p_alfa"] < NIVEL_SIG else "") for t in sel],
            textposition="top center", marker=dict(size=11, symbol=simb,
                                                   color=COR["mv"] if flag else COR["mercado"]),
            customdata=np.c_[reg.loc[sel, "alfa_aa"], reg.loc[sel, "p_alfa"]],
            hovertemplate="β=%{x:.2f}<br>retorno hist.=%{y:.2%}<br>α=%{customdata[0]:.2%} a.a."
                          "<br>p(α)=%{customdata[1]:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[1], y=[met.rf_aa + met.premio_hist_aa], mode="markers",
                             name="Mercado (β = 1)", marker=dict(size=12, symbol="x", color="black")))
    fig.update_layout(title="Linha do mercado de títulos (SML) — * = α significativo a 5%",
                      xaxis_title="β", yaxis_title="Retorno médio anual", yaxis_tickformat=".0%",
                      height=520, **LAYOUT)
    return fig


def fig_fronteira(res: ResultadoOtimizacao, met: ResultadoMetricas) -> go.Figure:
    a = res.aleatorias
    sig = np.sqrt(np.diag(met.cov_m.loc[res.mu.index, res.mu.index])) * np.sqrt(cfg.MESES_POR_ANO)
    tg, mv = res.tangente, res.min_var
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a["vol"], y=a["retorno"], mode="markers", name="Carteiras aleatórias",
                             marker=dict(size=3, color=a["sharpe"], colorscale="Viridis", opacity=0.5,
                                         colorbar=dict(title="Sharpe", x=1.02)),
                             hovertemplate="σ=%{x:.2%}<br>E[R]=%{y:.2%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=res.fronteira["vol"], y=res.fronteira["retorno"], mode="lines",
                             name="Fronteira eficiente", line=dict(color="black", width=3)))
    x_cml = np.array([0, max(tg.vol * 1.4, res.fronteira["vol"].max())])
    fig.add_trace(go.Scatter(x=x_cml, y=res.rf + tg.sharpe * x_cml, mode="lines", name="CML",
                             line=dict(color=COR["tg"], dash="dash")))
    fig.add_trace(go.Scatter(x=sig, y=res.mu, mode="markers+text", name="Ativos",
                             text=[nome(t) for t in res.mu.index], textposition="middle right",
                             marker=dict(size=9, color=COR["ativo"])))
    for c, cor, simb in [(mv, COR["mv"], "diamond"), (tg, COR["tg"], "star")]:
        fig.add_trace(go.Scatter(x=[c.vol], y=[c.retorno], mode="markers", name=c.nome,
                                 marker=dict(size=16, color=cor, symbol=simb,
                                             line=dict(width=1, color="black"))))
    fig.add_trace(go.Scatter(x=[0], y=[res.rf], mode="markers", name="r_f",
                             marker=dict(size=9, color="gray")))
    fig.update_layout(title=f"Fronteira eficiente e CML — retornos esperados: {res.cenario}",
                      xaxis_title="Volatilidade anual (σ)", yaxis_title="Retorno esperado anual",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=580,
                      legend=dict(orientation="h", y=-0.15), **LAYOUT)
    return fig


def fig_pesos(resultados: list[ResultadoOtimizacao]) -> go.Figure:
    series = {resultados[0].min_var.nome: resultados[0].min_var.pesos}
    for r in resultados:
        series[f"Tangente – {r.cenario}"] = r.tangente.pesos
    df = pd.DataFrame(series)
    fig = go.Figure([go.Bar(x=[nome(t) for t in df.index], y=df[c], name=c,
                            text=[pct(v, 0) if v > 0.005 else "" for v in df[c]], textposition="outside")
                     for c in df.columns])
    if cfg.PESO_MAX is not None:
        fig.add_hline(y=cfg.PESO_MAX, line_dash="dot", annotation_text=f"teto {pct(cfg.PESO_MAX, 0)}")
    fig.update_layout(barmode="group", title="Pesos das carteiras ótimas", yaxis_tickformat=".0%",
                      height=460, **LAYOUT)
    return fig


# ----------------------------------------------------------------------------
# Tabelas
# ----------------------------------------------------------------------------
def tabela_ativos(met: ResultadoMetricas) -> pd.DataFrame:
    reg, d = met.regressoes, met.descritivas
    cen0 = met.cenarios[0]
    linhas = {}
    for t in reg.index:
        linha = {
            "Retorno médio a.a.": pct(d.at[t, "media_aa"]),
            "Volatilidade a.a.": pct(d.at[t, "vol_aa"]),
            "β": num(reg.at[t, "beta"]),
            "IC 95% β": f"[{num(reg.at[t, 'beta_ic_inf'], 2)}; {num(reg.at[t, 'beta_ic_sup'], 2)}]",
            "α a.a.": pct(reg.at[t, "alfa_aa"]),
            "p(α)": num(reg.at[t, "p_alfa"]),
            "R²": num(reg.at[t, "r2"]),
            f"CAPM a.a. ({cen0.nome.lower()})": pct(cen0.retorno_capm_aa[t]),
            "Histórico − CAPM": pct(d.at[t, "media_aa"] - cen0.retorno_capm_aa[t]),
        }
        for cen in met.cenarios[1:]:
            linha[f"CAPM a.a. ({cen.nome.lower()})"] = pct(cen.retorno_capm_aa[t])
        linhas[nome(t)] = linha
    b = cfg.BENCHMARK
    linhas[f"{b} (mercado)"] = {"Retorno médio a.a.": pct(d.at[b, "media_aa"]),
                                "Volatilidade a.a.": pct(d.at[b, "vol_aa"]), "β": "1,000"}
    return pd.DataFrame(linhas).T.fillna("")


def tabela_carteiras(resultados: list[ResultadoOtimizacao]) -> pd.DataFrame:
    linhas = {}
    for r in resultados:
        for c in (r.min_var, r.tangente):
            if c is r.min_var and r is not resultados[0]:
                continue  # a variância mínima não depende de μ: mostrar só uma vez
            rot = c.nome if c is r.min_var else f"{c.nome} – {r.cenario}"
            linhas[rot] = {
                **{nome(t): pct(w, 1) for t, w in c.pesos.items()},
                # E[R] da variância mínima medido com μ histórico
                "E[R] a.a.": pct(c.retorno), "σ a.a.": pct(c.vol),
                "Sharpe": num(c.sharpe), "β carteira": num(c.beta),
            }
    return pd.DataFrame(linhas)


def tabela_comparativa(resultados: list[ResultadoOtimizacao], met: ResultadoMetricas) -> pd.DataFrame:
    h, c = resultados[0], resultados[1]
    df = pd.DataFrame({
        "μ histórico": h.mu.map(pct), "μ CAPM": c.mu.map(pct),
        "Peso tangente (histórico)": h.tangente.pesos.map(lambda v: pct(v, 1)),
        "Peso tangente (CAPM)": c.tangente.pesos.map(lambda v: pct(v, 1)),
        "Diferença (p.p.)": ((c.tangente.pesos - h.tangente.pesos) * 100).map(lambda v: num(v, 1)),
    })
    df.index = [nome(t) for t in df.index]
    return df


# ----------------------------------------------------------------------------
# Texto
# ----------------------------------------------------------------------------
def interpretacao(met: ResultadoMetricas, resultados: list[ResultadoOtimizacao]) -> str:
    """Parágrafo gerado a partir dos números. Revise antes de entregar."""
    reg, d = met.regressoes, met.descritivas
    h, c = resultados[0], resultados[1]
    tg = h.tangente.pesos.sort_values(ascending=False)
    dominantes = tg[tg >= PESO_RELEVANTE]
    sharpe_ind = ((d.loc[reg.index, "media_aa"] - met.rf_aa) / d.loc[reg.index, "vol_aa"])
    corr_media = met.corr.where(~np.eye(len(met.corr), dtype=bool)).mean()

    dom = list(dominantes.index)
    resto = [t for t in reg.index if t not in dom]
    no_teto = [nome(t) for t, w in tg.items() if cfg.PESO_MAX and w >= cfg.PESO_MAX - 1e-4]
    desc_dom = lista([f"{nome(t)} ({pct(w, 1)}; Sharpe individual {num(sharpe_ind[t], 2)}, "
                      f"correlação média {num(corr_media[t], 2)})" for t, w in dominantes.items()])
    acima = [nome(t) for t in reg.index if reg.at[t, "alfa_aa"] > 0]
    abaixo = [nome(t) for t in reg.index if reg.at[t, "alfa_aa"] <= 0]
    sig = [nome(t) for t in reg.index if reg.at[t, "p_alfa"] < NIVEL_SIG]
    r2 = reg["r2"]
    giro = 0.5 * (c.tangente.pesos - h.tangente.pesos).abs().sum()
    # Número efetivo de ativos = 1 / Σw² (inverso do índice de Herfindahl)
    n_ef_h = 1 / (h.tangente.pesos ** 2).sum()
    n_ef_c = 1 / (c.tangente.pesos ** 2).sum()
    if n_ef_c > n_ef_h + 0.5:
        comp = "mais diversificada"
    elif n_ef_c < n_ef_h - 0.5:
        comp = "mais concentrada"
    else:
        comp = "com grau de diversificação semelhante"

    p = []
    p.append(
        f"No período de {met.excessos.index[0]:%m/%Y} a {met.excessos.index[-1]:%m/%Y}, a carteira tangente "
        f"construída com os retornos médios históricos concentra-se em {desc_dom}"
        + (f", com {lista(no_teto)} limitada(s) pelo teto de {pct(cfg.PESO_MAX, 0)}" if no_teto else "")
        + f". Na lógica de Markowitz, um ativo ganha peso na tangente quando seu retorno em excesso é alto "
          f"em relação à volatilidade e quando sua correlação com os demais é baixa, pois isso reduz o risco "
          f"da carteira. As ações dominantes têm Sharpe individual médio de {num(sharpe_ind[dom].mean(), 2)}, "
          f"contra {num(sharpe_ind[resto].mean(), 2) if len(resto) else '—'} das demais, e correlação média "
          f"de {num(corr_media[dom].mean(), 2)}, contra "
          f"{num(corr_media[resto].mean(), 2) if len(resto) else '—'}. O Sharpe da carteira tangente é "
          f"{num(h.tangente.sharpe, 2)} e seu β é {num(h.tangente.beta, 2)}.")
    p.append(
        f"Em relação à SML estimada com prêmio histórico de {pct(met.premio_hist_aa)} a.a., ficam acima da "
        f"reta (α de Jensen positivo) {lista(acima)}, e abaixo {lista(abaixo)}. "
        + (f"Entretanto, apenas {lista(sig)} apresenta(m) alfa estatisticamente diferente de zero a 5%; "
           "para os demais, a distância à SML é compatível com ruído amostral e não permite afirmar "
           "que estejam sub ou sobreprecificados." if sig else
           "Entretanto, nenhum alfa é estatisticamente diferente de zero a 5%: as distâncias à SML são "
           "compatíveis com ruído amostral e não permitem afirmar que algum ativo esteja sub ou "
           "sobreprecificado, o que é consistente com o CAPM."))
    p.append(
        f"O R² das regressões varia de {num(r2.min(), 2)} a {num(r2.max(), 2)} (média {num(r2.mean(), 2)}): "
        f"o fator de mercado explica, em média, cerca de {pct(r2.mean(), 0)} da variância dos retornos mensais, "
        f"e o restante é risco específico (diversificável).")
    if met.premio_implausivel:
        p.append(
            f"O prêmio de mercado histórico do período ({pct(met.premio_hist_aa)} a.a.) é baixo ou negativo, "
            f"o que torna a SML histórica pouco informativa sobre retornos esperados; por isso, a otimização "
            f"com o CAPM usou o prêmio de referência de {pct(cfg.PREMIO_REFERENCIA_AA, 1)} a.a.")
    p.append(
        f"Por fim, substituir os retornos históricos pelos do CAPM altera a composição da carteira tangente "
        f"em {pct(giro, 0)} do patrimônio (metade da soma das diferenças absolutas de peso), evidenciando a "
        f"sensibilidade do modelo de Markowitz às estimativas de retorno esperado. Como o CAPM atribui a cada "
        f"ativo um retorno determinado apenas pelo seu β, ele elimina os alfas históricos; neste caso, a "
        f"carteira resultante ficou {comp} (número efetivo de ativos, 1/Σw², de {num(n_ef_h, 1)} com retornos "
        f"históricos e {num(n_ef_c, 1)} com o CAPM).")
    return " ".join(p)


LIMITACOES = [
    ("Retorno passado ≠ retorno esperado",
     "A média amostral de 60 meses é um estimador muito ruidoso do retorno esperado: com volatilidade "
     "anual de 30%, o erro-padrão da média anual é de cerca de 30%/√5 ≈ 13 p.p."),
    ("Sensibilidade de Markowitz aos inputs",
     "Pequenas mudanças em μ alteram muito os pesos ótimos (o otimizador amplifica erros de estimação); "
     "a comparação histórico × CAPM ilustra isso."),
    ("Instabilidade do beta",
     "O β é estimado com 60 observações e muda ao longo do tempo; o IC 95% na tabela indica a incerteza."),
    ("Normalidade e independência dos retornos",
     "A anualização por √12 e a análise média-variância supõem retornos i.i.d. e aproximadamente normais; "
     "retornos de ações têm caudas pesadas e assimetria. Os erros-padrão da regressão são OLS clássicos "
     "(sem correção para heterocedasticidade/autocorrelação)."),
    ("Taxa livre de risco",
     "A Selic mensal varia ao longo do período; usou-se a Selic de cada mês nos retornos em excesso e "
     "sua média anualizada como r_f na otimização e na SML."),
    ("Ausência de custos",
     "Não há custos de transação, impostos, rebalanceamento nem restrições de liquidez."),
    ("Viés de sobrevivência",
     "Os tickers foram escolhidos hoje entre empresas grandes e ainda listadas, o que tende a superestimar "
     "retornos passados."),
    ("Benchmark",
     "O Ibovespa é usado como proxy da carteira de mercado do CAPM, que na teoria inclui todos os ativos "
     "arriscados (Cap. 11.3)."),
]


# ----------------------------------------------------------------------------
# Montagem
# ----------------------------------------------------------------------------
CSS = """
body{font-family:Arial,sans-serif;max-width:1150px;margin:30px auto;padding:0 20px;color:#222;line-height:1.5}
h1{color:#1f4e79;border-bottom:3px solid #1f4e79;padding-bottom:6px}
h2{color:#1f4e79;margin-top:40px;border-bottom:1px solid #ccc;padding-bottom:4px}
.tab{border-collapse:collapse;font-size:13px;margin:12px 0;width:100%}
.tab th,.tab td{border:1px solid #ddd;padding:5px 8px;text-align:right}
.tab th{background:#f0f4f8}.tab tbody th{text-align:left}
.aviso{background:#fff4e5;border-left:5px solid #e65100;padding:10px 14px;margin:14px 0}
.nota{color:#555;font-size:13px}.resumo td{padding:3px 12px 3px 0}
.interp{background:#f6f9fc;border-left:5px solid #1f4e79;padding:12px 16px;text-align:justify}
"""


def gerar(dados: DadosMercado, met: ResultadoMetricas, resultados: list[ResultadoOtimizacao],
          destino: Path = cfg.ARQ_RELATORIO) -> Path:
    meta = dados.meta
    secoes: list[str] = []

    aviso_sint = ('<div class="aviso"><b>ATENÇÃO: dados sintéticos.</b> Este relatório foi gerado com dados '
                  'simulados para testar o código; os números não descrevem o mercado real.</div>'
                  if meta.get("sintetico") else "")
    aviso_premio = ""
    if met.premio_implausivel:
        aviso_premio = (
            f'<div class="aviso"><b>Prêmio de mercado histórico implausível:</b> a média de R_m − r_f no período foi '
            f'{pct(met.premio_hist_aa)} a.a. (abaixo do limiar de {pct(cfg.PREMIO_MINIMO_PLAUSIVEL_AA, 0)}). '
            f'Por isso, além do cenário <b>histórico</b>, foi rodado um cenário de <b>referência</b> com prêmio de '
            f'{pct(cfg.PREMIO_REFERENCIA_AA, 1)} a.a. A otimização "CAPM" usa o cenário de referência.</div>')

    resumo = {
        "Ativos": ", ".join(nome(t) for t in meta["ativos_usados"]),
        "Ativos descartados": ", ".join(meta.get("ativos_descartados", [])) or "nenhum",
        "Benchmark": cfg.BENCHMARK,
        "Período (retornos mensais)": f"{meta['primeiro_mes']} a {meta['ultimo_mes']} ({meta['n_meses']} meses)",
        "Fonte dos preços": meta.get("fonte_precos", "") + (
            f" — v{meta['yfinance_versao']}, auto_adjust={meta['auto_adjust']} (preços ajustados por proventos e desdobramentos)"
            if "yfinance_versao" in meta else ""),
        "Fonte de r_f": meta.get("fonte_rf", ""),
        "r_f médio": f"{pct(met.rf_aa)} a.a. ({pct(met.rf_aa / cfg.MESES_POR_ANO, 3)} a.m.)",
        "Prêmio de mercado histórico": f"{pct(met.premio_hist_aa)} a.a.",
        "Restrições": f"long-only, Σw = 1, teto por ativo = "
                      + ("nenhum" if cfg.PESO_MAX is None else pct(cfg.PESO_MAX, 0)),
        "Anualização": "média × 12; volatilidade × √12 (retornos mensais simples)",
    }
    if "baixado_em" in meta:
        resumo["Dados baixados em"] = meta["baixado_em"]
    secoes.append("<h2>1. Resumo</h2>" + aviso_sint + aviso_premio + '<table class="resumo">' + "".join(
        f"<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(str(v))}</td></tr>" for k, v in resumo.items())
        + "</table>")

    secoes.append(
        "<h2>2. Métricas por ativo</h2>" + tabela(tabela_ativos(met))
        + '<p class="nota">α = alfa de Jensen (intercepto da regressão), anualizado. p(α): p-valor do teste '
          'H₀: α = 0. Retorno CAPM = r_f + β (E[R_m] − r_f). No cenário histórico, “Histórico − CAPM” é igual a α '
          'por construção (a reta de MQO passa pelas médias).</p>')
    secoes.append("<h2>3. Correlação</h2>" + fig_html(fig_correlacao(met.corr)))
    secoes.append("<h2>4. Regressões do beta (Cap. 11.3)</h2>" + fig_html(fig_regressoes(met)))
    secoes.append(
        "<h2>5. Linha do mercado de títulos (Cap. 11.4)</h2>" + fig_html(fig_sml(met))
        + '<p class="nota">Pontos acima da SML tiveram retorno histórico maior que o exigido pelo CAPM para o '
          'seu β (α &gt; 0, “subprecificados” no período); abaixo, o contrário. O asterisco indica α '
          'significativo a 5%.</p>')

    sec6 = "<h2>6. Fronteira eficiente e CML</h2>"
    for r in resultados:
        sec6 += fig_html(fig_fronteira(r, met))
    secoes.append(sec6)

    secoes.append(
        "<h2>7. Carteiras ótimas</h2>" + fig_html(fig_pesos(resultados)) + tabela(tabela_carteiras(resultados))
        + "<h3>Comparação: retornos históricos × CAPM</h3>" + tabela(tabela_comparativa(resultados, met))
        + '<p class="nota">A carteira de variância mínima não depende dos retornos esperados e, por isso, é '
          'igual nos dois cenários.</p>')

    secoes.append("<h2>8. Limitações</h2><ul>" + "".join(
        f"<li><b>{html.escape(t)}.</b> {html.escape(x)}</li>" for t, x in LIMITACOES) + "</ul>")
    secoes.append(
        "<h2>9. Interpretação</h2>"
        + '<p class="nota">Texto gerado automaticamente a partir dos resultados; revise antes de entregar.</p>'
        + f'<div class="interp">{html.escape(interpretacao(met, resultados))}</div>')

    doc = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>Otimização de carteira — B3 (Markowitz + CAPM)</title>
<style>{CSS}</style>
<script type="text/javascript">{po.get_plotlyjs()}</script>
</head><body>
<h1>Otimização de carteira com ações da B3: Markowitz e CAPM</h1>
<p class="nota">Referencial: Berk, DeMarzo &amp; Harford, <i>Fundamentos de Finanças Empresariais</i>,
caps. 10 e 11. Gerado em {datetime.now():%d/%m/%Y %H:%M}.</p>
{''.join(secoes)}
</body></html>"""
    destino.write_text(doc, encoding="utf-8")
    return destino
