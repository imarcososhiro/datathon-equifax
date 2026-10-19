"""Interpretabilidade (Passo 9): contribuições SHAP e os gráficos do relatório.

[IA] Módulo gerado com auxílio de IA (Claude) em 13/10/2026. Registrar em docs/uso_ia.md.

As contribuições são exatas e vêm do próprio modelo:
- LightGBM e XGBoost: TreeSHAP nativo (`pred_contrib` / `pred_contribs`), que já
  lida com as variáveis categóricas como o modelo as usa;
- logística com WoE: coeficiente x WoE de cada variável (o modelo é aditivo).

Escala: log-odds de ADIMPLÊNCIA. Contribuição positiva aumenta o score (menos
risco); negativa diminui. Para cada cliente, valor_base + soma das contribuições
= log-odds do score. As VPs não têm significado conhecido, então a leitura é
"quanto e em que direção", sem história causal.

Gráficos que a banca de 2025 pediu para ler corretamente:
- barras: média do |SHAP| (quanto cada variável move o score, em média);
- beeswarm: direção e dispersão (cor = valor da variável, posição = efeito);
- dependência: como o efeito muda com o valor da variável;
- waterfall: por que um cliente recebeu aquele score;
- SHAP médio por safra: se a explicação é estável no tempo.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experimentos import preparar_x

ROTULO_ESCALA = "Contribuição SHAP (log-odds de adimplência; positivo = score maior)"


def contribuicoes(treinado, base, amostra=None, semente=0):
    """Contribuições de cada variável para cada linha.

    Devolve (contrib, valor_base, x): contrib é um DataFrame (linhas x variáveis),
    valor_base é um float e x é a matriz de entrada do modelo (mesmo índice).
    amostra: número de linhas sorteadas (None = todas).
    """
    if amostra is not None and amostra < len(base):
        base = base.sample(n=amostra, random_state=semente)
    x = preparar_x(base, treinado.variaveis, treinado.categoricas)
    est = treinado.estimador
    if treinado.modelo == "lgbm":
        valores = np.asarray(est.predict(x, pred_contrib=True))
    elif treinado.modelo == "xgb":
        import xgboost
        matriz = xgboost.DMatrix(x, enable_categorical=True)
        valores = est.get_booster().predict(matriz, pred_contribs=True)
    elif treinado.modelo == "logit_woe":
        woe = est.transformador_.transform(x)
        coef = est.coeficientes
        contrib = pd.DataFrame(0.0, index=x.index, columns=treinado.variaveis)
        contrib[coef.index] = woe[coef.index].to_numpy() * coef.to_numpy()
        return contrib, float(est.logistica_.intercept_[0]), x
    else:
        raise ValueError(f"Contribuições não implementadas para {treinado.modelo}")
    contrib = pd.DataFrame(valores[:, :-1], index=x.index, columns=treinado.variaveis)
    return contrib, float(valores[0, -1]), x


def conferir_soma(treinado, contrib, valor_base, x, tolerancia=1e-4):
    """Confere que valor_base + soma das contribuições = log-odds do score do modelo."""
    p = np.clip(treinado.estimador.predict_proba(x)[:, 1], 1e-12, 1 - 1e-12)
    logito = np.log(p / (1 - p))
    erro = np.abs(valor_base + contrib.sum(axis=1).to_numpy() - logito)
    return float(erro.max()) <= tolerancia, float(erro.max())


def importancia_shap(contrib):
    """Média do |SHAP| por variável, da maior para a menor."""
    return contrib.abs().mean().sort_values(ascending=False)


def shap_por_safra(contrib, safra, variaveis):
    """SHAP médio (com sinal) e média do |SHAP| de cada variável em cada safra."""
    safra = pd.Series(np.asarray(safra), index=contrib.index, name="safra")
    media = contrib[variaveis].groupby(safra).mean()
    media_abs = contrib[variaveis].abs().groupby(safra).mean()
    return media, media_abs


def escolher_perfis(score, quantis=(0.05, 0.5, 0.95), indice=None):
    """Índices dos clientes mais perto de cada quantil do score (risco alto, médio, baixo)."""
    score = pd.Series(np.asarray(score, dtype=float), index=indice)
    alvo = score.quantile(list(quantis))
    return {f"q{int(q * 100):02d}": (score - v).abs().idxmin() for q, v in zip(quantis, alvo)}


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def _numerica(x):
    """Matriz numérica para colorir os gráficos (categorias viram o código da categoria)."""
    saida = x.copy()
    for c in saida.columns:
        if not pd.api.types.is_numeric_dtype(saida[c]):
            saida[c] = saida[c].astype("category").cat.codes.replace(-1, np.nan)
    return saida.astype(float)


def grafico_importancia(contrib, n=20, titulo="Importância SHAP"):
    imp = importancia_shap(contrib).head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * n + 1.5))
    ax.barh(imp.index, imp.values, color="steelblue")
    ax.set_xlabel("Média do |SHAP| (log-odds)")
    ax.set_title(titulo)
    ax.grid(axis="x", alpha=.3)
    return fig


def grafico_beeswarm(contrib, valor_base, x, n=15, titulo="Direção e dispersão do efeito"):
    import shap
    variaveis = list(importancia_shap(contrib).index[:n])
    exp = shap.Explanation(values=contrib[variaveis].to_numpy(),
                           base_values=np.full(len(contrib), valor_base),
                           data=_numerica(x[variaveis]).to_numpy(), feature_names=variaveis)
    plt.figure()  # a função do shap desenha na figura aberta: começa numa nova
    shap.plots.beeswarm(exp, max_display=n, show=False)
    fig = plt.gcf()
    fig.axes[0].set_xlabel(ROTULO_ESCALA)
    fig.axes[0].set_title(titulo)
    return fig


def grafico_dependencia(contrib, x, variavel, codigos_especiais=(-1, -2, 9999)):
    """Efeito x valor da variável.

    Numérica: pontos para os valores normais; cada código especial presente aparece
    à esquerda do eixo, numa posição própria, com o efeito médio escrito.
    Categórica: barra com a contribuição média de cada categoria.
    """
    valor = x[variavel]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not pd.api.types.is_numeric_dtype(valor):
        medias = contrib[variavel].groupby(valor.astype(str), observed=True).mean().sort_values()
        ax.barh(medias.index, medias.values, color="steelblue")
        ax.axvline(0, color="gray", lw=.8)
        ax.set_xlabel("Contribuição SHAP média (log-odds de adimplência)")
        ax.set_title(f"Dependência: {variavel}")
        ax.grid(axis="x", alpha=.3)
        return fig

    especial = valor.isin(codigos_especiais)
    normais = valor[~especial & valor.notna()]
    ax.scatter(normais, contrib.loc[normais.index, variavel], s=4, alpha=.3, color="steelblue")
    presentes = sorted(valor[especial].unique())
    if presentes:
        inicio = normais.min() if len(normais) else 0.0
        largura = (normais.max() - normais.min()) if len(normais) > 1 else 1.0
        rotulos, posicoes = [], []
        for k, codigo in enumerate(presentes, 1):
            m = valor == codigo
            pos = inicio - largura * 0.08 * k
            ax.scatter(np.full(m.sum(), pos), contrib.loc[m, variavel], s=6, alpha=.4, color="crimson")
            rotulos.append(f"{codigo:g}")
            posicoes.append(pos)
            ax.annotate(f"{contrib.loc[m, variavel].mean():+.2f}", (pos, contrib.loc[m, variavel].mean()),
                        textcoords="offset points", xytext=(4, 0), color="crimson", fontsize=8)
        ticks = list(ax.get_xticks())
        ax.set_xticks(posicoes + [t for t in ticks if t >= inicio])
        ax.set_xticklabels(rotulos + [f"{t:g}" for t in ticks if t >= inicio])
        ax.set_xlim(min(posicoes) - largura * 0.05, None)
    ax.axhline(0, color="gray", lw=.8)
    ax.set_xlabel(f"{variavel} (em vermelho: códigos especiais)" if presentes else variavel)
    ax.set_ylabel(ROTULO_ESCALA)
    ax.set_title(f"Dependência: {variavel}")
    ax.grid(alpha=.3)
    return fig


def grafico_waterfall(contrib, valor_base, x, indice, n=12, titulo=None):
    """Do valor base ao score do cliente, somando a contribuição de cada variável.

    O eixo está em log-odds; o título traz a probabilidade correspondente.
    As categóricas aparecem com o rótulo (ex.: 'Bahia'), não com o código.
    """
    import shap
    linha = contrib.loc[indice]
    valores = x.loc[indice]
    rotulos = np.array([v if not isinstance(v, (int, float, np.number)) or pd.isna(v) else
                        (f"{v:g}" if isinstance(v, (float, np.floating)) else v)
                        for v in valores.to_numpy()], dtype=object)
    exp = shap.Explanation(values=linha.to_numpy(), base_values=valor_base, data=rotulos,
                           feature_names=list(contrib.columns))
    plt.figure()  # a função do shap desenha na figura aberta: começa numa nova
    shap.plots.waterfall(exp, max_display=n, show=False)
    fig = plt.gcf()
    logito = valor_base + linha.sum()
    probabilidade = 1 / (1 + np.exp(-logito))
    fig.axes[0].set_title((titulo + "\n" if titulo else "") +
                          f"log-odds {logito:.2f} = probabilidade de adimplência {probabilidade:.3f}",
                          fontsize=10)
    return fig


def grafico_shap_por_safra(media_abs, titulo="Média do |SHAP| por safra"):
    fig, ax = plt.subplots(figsize=(9, 5))
    for v in media_abs.columns:
        ax.plot(media_abs.index.astype(str), media_abs[v], marker="o", label=v)
    ax.set_xlabel("Safra")
    ax.set_ylabel("Média do |SHAP| (log-odds)")
    ax.set_title(titulo)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, ncol=2)
    return fig
