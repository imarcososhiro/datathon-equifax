"""Regressão logística com WoE (scorecard) para o motor de experimentos.

[IA] Módulo gerado com auxílio de IA (Claude) em 12/10/2026. Registrar em docs/uso_ia.md.

Cada variável é trocada pelo WoE da faixa em que cai, com o binning ótimo de
`selecao.ajustar_binning` (o mesmo da ficha), ajustado SÓ nos dados recebidos
no `fit` (o treino, ou o treino de cada dobra). Depois:

1. saem as variáveis com IV abaixo de `iv_minimo`;
2. entre as que sobram, sai uma de cada grupo com |correlação| >= `limiar_correlacao`
   entre os WoE (fica a de maior IV, como na seleção do Passo 4);
3. uma regressão logística regularizada é ajustada nos WoE:
   `l1_ratio=0` é L2 (ridge), `l1_ratio=1` é L1 (lasso). Desde o scikit-learn 1.8
   o parâmetro `penalty` está obsoleto; a escolha é feita por `l1_ratio`.

Sem dummies e com regularização: são os dois problemas que a banca de 2025
apontou na logística daquele grupo (não convergiu).

Atenção ao optbinning: por padrão, `transform` devolve WoE 0 para os códigos
especiais e para o nulo. Aqui usamos `metric_special="empirical"` e
`metric_missing="empirical"`, para que o -1, o -2 e o 9999 recebam o WoE da
faixa própria deles. Categoria que não apareceu no fit recebe WoE 0 (neutro).

Convenção do WoE: ln(%bons / %maus); WoE > 0 indica faixa com mais bons.
"""
import copy
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .selecao import ajustar_binning, iv_de, selecionar_por_correlacao, tabela_woe

_OPCOES_TRANSFORM = dict(metric="woe", metric_special="empirical", metric_missing="empirical")


def _e_categorica(serie):
    return not pd.api.types.is_numeric_dtype(serie)


def _como_array(serie):
    """Numéricas em float; categóricas em texto com o nulo como NaN."""
    if _e_categorica(serie):
        valores = serie.astype(object).to_numpy()
        return np.array([v if pd.notna(v) else np.nan for v in valores], dtype=object)
    return serie.to_numpy(dtype="float64")


def _ajustar_uma(nome, x, y):
    """Binning de uma variável; devolve (nome, binning ou None, IV, erro)."""
    tipo = "categorical" if x.dtype == object else "numerical"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ob = ajustar_binning(pd.Series(x, name=nome), pd.Series(y), tipo=tipo)
        return nome, ob, iv_de(ob), None
    except Exception as erro:
        return nome, None, np.nan, str(erro)[:200]


class TransformadorWoE(BaseEstimator):
    """Ajusta o binning de cada coluna e devolve a matriz de WoE das que ficam."""

    def __init__(self, iv_minimo=0.02, limiar_correlacao=0.8, n_jobs=-1):
        self.iv_minimo = iv_minimo
        self.limiar_correlacao = limiar_correlacao
        self.n_jobs = n_jobs

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        colunas = list(X.columns)
        resultados = Parallel(n_jobs=self.n_jobs)(
            delayed(_ajustar_uma)(c, _como_array(X[c]), y) for c in colunas)

        self._binnings_iv, self._removidas_iv = {}, {}
        ivs = {}
        for nome, ob, iv, erro in resultados:
            ivs[nome] = iv
            if ob is None:
                self._removidas_iv[nome] = f"erro no binning: {erro}"
            elif iv < self.iv_minimo:
                self._removidas_iv[nome] = f"IV {iv:.4f} < {self.iv_minimo}"
            else:
                self._binnings_iv[nome] = ob
        self.iv_ = pd.Series(ivs, dtype=float)

        # Correlação entre os WoE das que passaram no IV (guardada para com_limiar)
        self._candidatas = [c for c in colunas if c in self._binnings_iv]
        self._corr = None
        if len(self._candidatas) > 1:
            self.binnings_ = self._binnings_iv
            woe = self._woe(X, self._candidatas)
            self._corr = pd.DataFrame(
                np.nan_to_num(np.corrcoef(woe.to_numpy(), rowvar=False)),
                index=self._candidatas, columns=self._candidatas)
        self._aplicar_limiar(self.limiar_correlacao)
        return self

    def _aplicar_limiar(self, limiar):
        self.binnings_ = dict(self._binnings_iv)
        self.removidas_ = dict(self._removidas_iv)
        if limiar is not None and limiar < 1 and self._corr is not None:
            ficam, saem = selecionar_por_correlacao(self._corr, self.iv_[self._candidatas], limiar)
            for sai, fica in saem.items():
                self.removidas_[sai] = (f"correlação {self._corr.loc[sai, fica]:.3f} com {fica} "
                                        f"(limiar {limiar})")
                del self.binnings_[sai]
        self.variaveis_ = [c for c in self._candidatas if c in self.binnings_]
        if not self.variaveis_:
            raise ValueError("Nenhuma variável sobrou depois dos filtros de IV e correlação")

    def com_limiar(self, limiar):
        """Cópia com outro limiar de correlação, sem refazer o binning."""
        novo = copy.copy(self)
        novo.limiar_correlacao = limiar
        novo._aplicar_limiar(limiar)
        return novo

    def _woe(self, X, variaveis):
        return pd.DataFrame(
            {c: self.binnings_[c].transform(_como_array(X[c]), **_OPCOES_TRANSFORM)
             for c in variaveis}, index=X.index).astype("float64")

    def transform(self, X):
        return self._woe(X, self.variaveis_)


class LogisticaWoE(BaseEstimator, ClassifierMixin):
    """Logística regularizada sobre WoE; interface de classificador do scikit-learn.

    `fit(X, y, transformador=...)` aceita um TransformadorWoE já ajustado nos
    mesmos dados, para a busca de C por validação cruzada não refazer o binning.
    """

    def __init__(self, C=1.0, l1_ratio=0.0, iv_minimo=0.02, limiar_correlacao=0.8,
                 max_iter=2000, random_state=None, n_jobs=-1):
        self.C = C
        self.l1_ratio = l1_ratio
        self.iv_minimo = iv_minimo
        self.limiar_correlacao = limiar_correlacao
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None, transformador=None):
        y = np.asarray(y).astype(int)
        self.colunas_ = list(X.columns)
        self.transformador_ = transformador or TransformadorWoE(
            self.iv_minimo, self.limiar_correlacao, self.n_jobs).fit(X, y)
        woe = self.transformador_.transform(X)
        solver = "lbfgs" if self.l1_ratio == 0 else "saga"
        self.logistica_ = LogisticRegression(C=self.C, l1_ratio=self.l1_ratio, solver=solver,
                                             max_iter=self.max_iter,
                                             random_state=self.random_state)
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always", ConvergenceWarning)
            self.logistica_.fit(woe.to_numpy(), y, sample_weight=sample_weight)
        self.convergiu_ = not any(issubclass(a.category, ConvergenceWarning) for a in avisos)
        if not self.convergiu_:
            warnings.warn("A logística não convergiu; aumente max_iter ou C menor",
                          ConvergenceWarning)
        self.classes_ = self.logistica_.classes_
        self._desvio_woe = woe.std().to_numpy()
        return self

    def predict_proba(self, X):
        return self.logistica_.predict_proba(self.transformador_.transform(X).to_numpy())

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    @property
    def coeficientes(self):
        """Coeficiente de cada variável que entrou na logística (0 = zerada pelo L1)."""
        return pd.Series(self.logistica_.coef_[0], index=self.transformador_.variaveis_)

    @property
    def feature_importances_(self):
        """|coeficiente| x desvio do WoE, para todas as colunas recebidas (0 = saiu)."""
        efeito = pd.Series(np.abs(self.logistica_.coef_[0]) * self._desvio_woe,
                           index=self.transformador_.variaveis_)
        return efeito.reindex(self.colunas_, fill_value=0.0).to_numpy()


def tabela_pontos(modelo, pdo=20, pontos_base=600, odds_base=50):
    """Scorecard: pontos de cada faixa de cada variável.

    Escala usual de mercado: `pontos_base` pontos correspondem a uma chance de
    `odds_base` bons para 1 mau, e a chance dobra a cada `pdo` pontos.
    O score de um cliente é a soma dos pontos das faixas em que ele cai e é igual a
    pontos_base + pdo/ln(2) * (ln(chance de ser bom) - ln(odds_base)).
    Variáveis com coeficiente zero (L1) ficam de fora.
    """
    fator = pdo / np.log(2)
    deslocamento = pontos_base - fator * np.log(odds_base)
    coef = modelo.coeficientes
    coef = coef[coef != 0]
    intercepto = float(modelo.logistica_.intercept_[0])
    por_variavel = (fator * intercepto + deslocamento) / len(coef)
    partes = []
    for var, c in coef.items():
        t = tabela_woe(modelo.transformador_.binnings_[var])[["Bin", "total", "taxa_inadimplencia", "WoE"]].copy()
        t["WoE"] = t["WoE"].astype(float)
        t.insert(0, "variavel", var)
        t["coeficiente"] = c
        t["pontos"] = (fator * c * t["WoE"] + por_variavel).round(1)
        partes.append(t)
    return pd.concat(partes, ignore_index=True)
