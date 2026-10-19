"""Divisão temporal e agrupada da base de desenvolvimento, e métricas de avaliação.

[IA] Módulo gerado com auxílio de IA (Claude). Registrar em docs/uso_ia.md.

Desenho (Passo 1):
    202301-202308 -> treino + teste, divididos agrupando por id
    202309-202310 -> OOT interna (só confirma; não entra em nenhuma decisão)

Convenção: adimplencia = 1 é adimplente; score maior = risco menor.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .config import RANDOM_SEED

SAFRAS_DEV = range(202301, 202309)      # 202301..202308
SAFRAS_OOT_INTERNA = (202309, 202310)


def separar_janelas(df, col_safra="safra"):
    """Separa a base em (dev, oot_interna) pelas safras."""
    safra = df[col_safra].astype(int)
    dev = df[safra.isin(SAFRAS_DEV)]
    oot_interna = df[safra.isin(SAFRAS_OOT_INTERNA)]
    return dev, oot_interna


def dividir_treino_teste(dev, col_target="adimplencia", col_grupo="id",
                         prop_teste=0.25, seed=RANDOM_SEED):
    """Divide dev em treino e teste sem que um mesmo id caia nos dois lados.

    Usa StratifiedGroupKFold para manter a taxa de inadimplência parecida
    nos dois conjuntos. prop_teste=0.25 equivale a 4 folds (1 vira teste).
    """
    n_splits = int(round(1 / prop_teste))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx_treino, idx_teste = next(sgkf.split(dev, dev[col_target], groups=dev[col_grupo]))
    treino, teste = dev.iloc[idx_treino], dev.iloc[idx_teste]

    # Garantia: nenhum id em comum
    assert not set(treino[col_grupo]) & set(teste[col_grupo]), "id nos dois lados"
    return treino, teste


def _metricas(y, score):
    """AUC, Gini e KS de um conjunto. Devolve NaN se só houver uma classe."""
    base = {"n": len(y), "taxa_inad": float((y == 0).mean()) if len(y) else np.nan}
    if len(np.unique(y)) < 2:
        return {**base, "auc": np.nan, "gini": np.nan, "ks": np.nan}
    auc = roc_auc_score(y, score)
    ks = ks_2samp(score[y == 1], score[y == 0]).statistic
    return {**base, "auc": auc, "gini": 2 * auc - 1, "ks": ks}


def avaliar(y, score, safra):
    """Calcula AUC, Gini e KS no total e por safra.

    y: 1 = adimplente, 0 = inadimplente.
    score: quanto maior, menor o risco (ex.: probabilidade de adimplência).
    Devolve um DataFrame com a linha "todas" seguida de uma linha por safra.
    """
    y, score, safra = np.asarray(y), np.asarray(score, dtype=float), np.asarray(safra)
    if not (len(y) == len(score) == len(safra)):
        raise ValueError("y, score e safra precisam ter o mesmo tamanho")
    if np.isnan(score).any():
        raise ValueError("score contém NaN")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y deve conter apenas 0 e 1")

    linhas = [{"safra": "todas", **_metricas(y, score)}]
    for s in np.unique(safra):
        m = safra == s
        linhas.append({"safra": str(s), **_metricas(y[m], score[m])})
    return pd.DataFrame(linhas)
