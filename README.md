# Otimizador de carteira — B3 (Markowitz + CAPM)

Trabalho 2 de Administração Financeira (Gestão de Investimentos). Referencial: Berk, DeMarzo & Harford,
*Fundamentos de Finanças Empresariais*, caps. 10 e 11.

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                # baixa os dados (1ª vez), salva em data/ e gera relatorio.html
python main.py --atualizar    # força novo download
python main.py --sintetico    # testa sem rede, com dados simulados (gera relatorio_sintetico.html)
```

Depois do primeiro download, a análise roda **sem rede** a partir de `data/` (o cache só é usado se os
tickers e o período em `config.py` forem os mesmos).

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `config.py` | Todos os parâmetros (tickers, período, teto, prêmio de referência, tolerâncias) |
| `data.py` | Coleta (yfinance + API SGS/BCB), cache em CSV, tratamento de ausências, dados sintéticos |
| `metrics.py` | Estatísticas descritivas, beta por OLS, CAPM/SML |
| `optimize.py` | Variância mínima, fronteira eficiente, carteira tangente, carteiras aleatórias, validações |
| `report.py` | Gráficos Plotly e `relatorio.html` autocontido |
| `main.py` | Orquestração |

## Decisões não óbvias

- **Frequência mensal:** convenção do Cap. 11.3 para estimar o β (5 anos de retornos mensais). O preço do
  mês é o último fechamento ajustado do mês; o mês anterior ao início é baixado só como base do 1º retorno.
- **Preços ajustados:** `auto_adjust=True` (passado explicitamente e registrado em `data/metadados.json`),
  então o `Close` já incorpora desdobramentos e proventos. Retornos mensais acima de 50% em módulo vão
  para o log como possível desdobramento não ajustado.
- **Anualização aritmética:** média × 12 e volatilidade × √12 (supõe retornos mensais i.i.d.). É
  consistente com o uso de médias aritméticas como retorno esperado no CAPM e em Markowitz.
- **Selic:** série SGS 4390 (Selic acumulada no mês, % a.m.), dividida por 100. A Selic de cada mês entra
  no retorno em excesso daquele mês; a média anualizada (× 12) é o r_f da SML e da otimização. Se os
  valores ficarem fora de 0–3% a.m., o código para com erro (código ou unidade errados).
- **Amostra comum:** ativos com menos de 90% dos meses são descartados; dos restantes, usam-se só os meses
  em que todos têm retorno, para que a matriz de covariância seja coerente (e PSD).
- **Prêmio de mercado:** média histórica de R_m − r_f. Se ficar abaixo de 2% a.a., o relatório sinaliza e
  roda também o cenário de referência (5% a.a.), que passa a ser o usado na otimização "CAPM" (com prêmio
  ≤ 0 a carteira tangente nem existe).
- **Solver SLSQP:** aceita limites por ativo e restrições de igualdade ao mesmo tempo; a variância mínima é
  um QP convexo. A tangente parte do ponto de maior Sharpe da fronteira, evitando ótimos locais.
- **Validação da variância mínima com teto:** com teto de 30%, investir 100% num único ativo não é viável,
  então a regra "σ_mv ≤ min σᵢ" não se aplica. Nesse caso o código verifica σ_mv ≤ menor σ entre as
  carteiras viáveis amostradas (igual-ponderada e as 5.000 aleatórias). Sem teto, usa a regra original.
- **Consistência alfa × SML:** como a reta de MQO passa pelas médias, no cenário histórico
  "retorno médio − retorno CAPM" = 12 × α exatamente; o código checa isso.

## Testes feitos (dados sintéticos)

- Modelo de um fator com βs conhecidos: em 200 simulações, o IC 95% do β cobriu o valor verdadeiro em 96%
  dos casos.
- Caminhos de prêmio plausível e implausível, com e sem teto; erros esperados para teto inviável e matriz
  não PSD; coleta e cache testados com respostas simuladas do yfinance e da API do BCB.
