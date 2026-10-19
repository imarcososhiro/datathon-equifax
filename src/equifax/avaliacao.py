"""Avaliação de um score além de AUC, Gini e KS: decis, estabilidade, calibração e política.

[IA] Módulo gerado com auxílio de IA (Claude) em 12/10/2026. Registrar em docs/uso_ia.md.

Serve aos três modelos (M1, M2, M3). Convenção do projeto: y = 1 é adimplente e
o score é a probabilidade de adimplência (score maior = risco menor). Tudo aqui
recebe arrays já calculados; nenhuma função treina ou escolhe nada.

Decil 1 = os 10% de clientes com o MENOR score (maior risco).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_curve

from .selecao import _psi_por_safra
from .validacao import avaliar


def _validar(y, score):
    y, score = np.asarray(y).astype(int), np.asarray(score, dtype=float)
    if len(y) != len(score):
        raise ValueError("y e score precisam ter o mesmo tamanho")
    if np.isnan(score).any():
        raise ValueError("score contém NaN")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y deve conter apenas 0 e 1")
    return y, score


# --------------------------------------------------------------------------- #
# Tabelas
# --------------------------------------------------------------------------- #
def tabela_decis(y, score, n_faixas=10):
    """Uma linha por faixa de score, do maior risco (faixa 1) para o menor.

    Colunas: score mínimo e máximo, clientes, maus, taxa de inadimplência da
    faixa, taxa acumulada (da faixa 1 até ela), % dos clientes e % dos maus
    acumulados até a faixa (quantos maus seriam barrados recusando até ali).
    Empates de score são separados pela ordem (rank 'first'), para as faixas
    terem o mesmo tamanho.
    """
    y, score = _validar(y, score)
    ordem = pd.Series(score).rank(method="first")
    faixa = pd.qcut(ordem, n_faixas, labels=range(1, n_faixas + 1)).astype(int)
    df = pd.DataFrame({"faixa": faixa, "score": score, "mau": 1 - y})
    t = df.groupby("faixa").agg(score_min=("score", "min"), score_max=("score", "max"),
                                clientes=("mau", "size"), maus=("mau", "sum"))
    t["taxa_inad"] = t["maus"] / t["clientes"]
    t["taxa_inad_acumulada"] = t["maus"].cumsum() / t["clientes"].cumsum()
    t["pct_clientes_acum"] = t["clientes"].cumsum() / t["clientes"].sum()
    t["pct_maus_capturados"] = t["maus"].cumsum() / max(t["maus"].sum(), 1)
    return t


def psi_score_por_safra(score, safra, safras_ref=None, n_faixas=10):
    """PSI do score de cada safra contra as primeiras safras (padrão: as 2 primeiras).

    Usa o mesmo cálculo do PSI das variáveis (`selecao`), com faixas por decil
    da referência e sem códigos especiais.
    """
    score = pd.Series(np.asarray(score, dtype=float), name="score")
    safra = pd.Series(np.asarray(safra))
    return _psi_por_safra(score, safra, safras_ref, n_faixas, codigos_especiais=[])


def tabela_calibracao(y, score, n_faixas=10):
    """Inadimplência prevista (1 - score) contra a observada, por decil, e o Brier.

    Só faz sentido se o score for uma probabilidade; um modelo treinado com
    peso de classe ordena bem, mas fica descalibrado.
    Devolve (tabela, brier), com o Brier calculado para o evento "mau".
    """
    y, score = _validar(y, score)
    prev = 1 - score
    ordem = pd.Series(prev).rank(method="first")
    faixa = pd.qcut(ordem, n_faixas, labels=range(1, n_faixas + 1)).astype(int)
    df = pd.DataFrame({"faixa": faixa, "prevista": prev, "observada": 1 - y})
    t = df.groupby("faixa").agg(clientes=("observada", "size"),
                                inad_prevista=("prevista", "mean"),
                                inad_observada=("observada", "mean"))
    return t, float(brier_score_loss(1 - y, prev))


def simular_politica(y, score, taxas_aprovacao=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """Aprova os clientes de maior score até a taxa de aprovação; mede o resultado.

    Para cada taxa: score de corte, aprovados, inadimplência entre os aprovados,
    inadimplência entre os recusados e % dos maus que foram recusados.
    """
    y, score = _validar(y, score)
    ordem = np.argsort(-score, kind="stable")
    y_ord, s_ord = y[ordem], score[ordem]
    n, total_maus = len(y), max(int((y == 0).sum()), 1)
    linhas = []
    for taxa in taxas_aprovacao:
        k = int(round(taxa * n))
        aprovados, recusados = y_ord[:k], y_ord[k:]
        linhas.append({
            "taxa_aprovacao": taxa,
            "score_corte": float(s_ord[k - 1]) if k else np.nan,
            "aprovados": k,
            "inad_aprovados": float((aprovados == 0).mean()) if k else np.nan,
            "inad_recusados": float((recusados == 0).mean()) if len(recusados) else np.nan,
            "pct_maus_recusados": float((recusados == 0).sum() / total_maus),
        })
    return pd.DataFrame(linhas).set_index("taxa_aprovacao")


# --------------------------------------------------------------------------- #
# Gráficos (devolvem a figura; quem chama decide se salva)
# --------------------------------------------------------------------------- #
def grafico_ks_por_safra(metricas, titulo="KS por safra"):
    """metricas: dict nome do modelo -> saída de `validacao.avaliar` (com a linha 'todas')."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for nome, m in metricas.items():
        por_safra = m[m["safra"] != "todas"]
        ax.plot(por_safra["safra"].astype(str), por_safra["ks"], marker="o", label=nome)
    ax.set_xlabel("Safra")
    ax.set_ylabel("KS")
    ax.set_title(titulo)
    ax.grid(color="gray", alpha=.3)
    ax.legend()
    return fig


def grafico_distribuicao_score(score, safra, titulo="Distribuição do score por safra"):
    """Um boxplot do score por safra: mostra se a distribuição muda no tempo."""
    df = pd.DataFrame({"score": np.asarray(score, dtype=float), "safra": np.asarray(safra)})
    safras = sorted(df["safra"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([df.loc[df["safra"] == s, "score"] for s in safras], showfliers=False)
    ax.set_xticks(range(1, len(safras) + 1), [str(s) for s in safras])
    ax.set_xlabel("Safra")
    ax.set_ylabel("Score (probabilidade de adimplência)")
    ax.set_title(titulo)
    ax.grid(axis="y", color="gray", alpha=.3)
    return fig


def grafico_roc(curvas, titulo="Curva ROC"):
    """curvas: dict nome -> (y, score)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for nome, (y, score) in curvas.items():
        y, score = _validar(y, score)
        fpr, tpr, _ = roc_curve(y, score)
        auc = avaliar(y, score, np.zeros(len(y))).loc[0, "auc"]
        ax.plot(fpr, tpr, label=f"{nome} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("Taxa de falsos positivos (maus aprovados)")
    ax.set_ylabel("Taxa de verdadeiros positivos (bons aprovados)")
    ax.set_title(titulo)
    ax.legend(loc="lower right")
    return fig


def grafico_calibracao(y, score, titulo="Calibração"):
    tabela, brier = tabela_calibracao(y, score)
    fig, ax = plt.subplots(figsize=(6, 6))
    limite = max(tabela["inad_prevista"].max(), tabela["inad_observada"].max()) * 1.05
    ax.plot([0, limite], [0, limite], color="gray", linestyle="--")
    ax.plot(tabela["inad_prevista"], tabela["inad_observada"], marker="o")
    ax.set_xlabel("Inadimplência prevista (média do decil)")
    ax.set_ylabel("Inadimplência observada")
    ax.set_title(f"{titulo} (Brier {brier:.4f})")
    ax.grid(color="gray", alpha=.3)
    return fig
