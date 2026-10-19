"""Testes de src/equifax/modelos.py (logística com WoE) com dados inventados.

[IA] Testes gerados com auxílio de IA (Claude) em 12/10/2026. Registrar em docs/uso_ia.md.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from equifax.modelos import LogisticaWoE, TransformadorWoE, tabela_pontos
from equifax.selecao import tabela_woe


@pytest.fixture
def dados():
    """VP01 forte com código -1 informativo; VP02 cópia de VP01; VP03 ruído; categórica."""
    rng = np.random.default_rng(1)
    n = 6000
    vp01 = rng.normal(size=n)
    especial = rng.random(n) < 0.15
    vp01[especial] = -1
    estado = rng.choice(["Paraná", "Bahia", "São Paulo"], n)
    logito = 1.8 + 0.9 * np.where(especial, 1.5, vp01) - 0.8 * (estado == "Bahia")
    y = (rng.random(n) < 1 / (1 + np.exp(-logito))).astype(int)
    x = pd.DataFrame({"VP01": vp01, "VP02": vp01 * 10 + 3, "VP03": rng.normal(size=n),
                      "estado": pd.Categorical(estado)})
    return x, y


def test_codigo_especial_recebe_o_woe_da_propria_faixa(dados):
    x, y = dados
    t = TransformadorWoE(iv_minimo=0, limiar_correlacao=None, n_jobs=1).fit(x, y)
    tabela = tabela_woe(t.binnings_["VP01"])
    woe_tabela = float(tabela.loc[tabela["Bin"].astype(str) == "esp_-1", "WoE"].iloc[0])
    woe = t.transform(x)["VP01"]
    assert woe[x["VP01"] == -1].iloc[0] == pytest.approx(woe_tabela)
    assert woe_tabela > 0.3  # o -1 indica menos risco nesta base inventada


def test_filtros_de_iv_e_correlacao(dados):
    x, y = dados
    t = TransformadorWoE(iv_minimo=0.02, limiar_correlacao=0.8, n_jobs=1).fit(x, y)
    assert "VP03" in t.removidas_ and "IV" in t.removidas_["VP03"]
    # VP02 é a mesma informação de VP01: só uma das duas fica
    assert ("VP01" in t.variaveis_) != ("VP02" in t.variaveis_)
    assert "correlação" in next(v for k, v in t.removidas_.items() if k in ("VP01", "VP02"))
    assert "estado" in t.variaveis_


def test_categoria_nova_vira_woe_zero(dados):
    x, y = dados
    t = TransformadorWoE(iv_minimo=0, limiar_correlacao=None, n_jobs=1).fit(x, y)
    novo = x.head(2).copy()
    novo["estado"] = pd.Categorical(["Acre", np.nan])
    assert np.allclose(t.transform(novo)["estado"], 0)


def test_logistica_ordena_e_converge(dados):
    x, y = dados
    m = LogisticaWoE(n_jobs=1).fit(x.iloc[:4000], y[:4000])
    assert m.convergiu_
    assert list(m.classes_) == [0, 1]
    p = m.predict_proba(x.iloc[4000:])[:, 1]
    assert roc_auc_score(y[4000:], p) > 0.7
    imp = pd.Series(m.feature_importances_, index=x.columns)
    assert imp["VP03"] == 0 and imp[m.transformador_.variaveis_].gt(0).all()


def test_l1_com_c_pequeno_zera_coeficientes(dados):
    x, y = dados
    m = LogisticaWoE(C=0.0005, l1_ratio=1.0, iv_minimo=0, limiar_correlacao=None,
                     n_jobs=1, random_state=0).fit(x, y)
    assert (m.coeficientes == 0).any()


def test_reaproveita_transformador(dados):
    x, y = dados
    t = TransformadorWoE(n_jobs=1).fit(x, y)
    a = LogisticaWoE(C=0.1, n_jobs=1).fit(x, y, transformador=t)
    b = LogisticaWoE(C=0.1, n_jobs=1).fit(x, y)
    np.testing.assert_allclose(a.predict_proba(x), b.predict_proba(x))


def test_pontos_somam_o_score(dados):
    x, y = dados
    m = LogisticaWoE(n_jobs=1).fit(x, y)
    pontos = tabela_pontos(m, pdo=20, pontos_base=600, odds_base=50)
    assert set(pontos["variavel"]) == set(m.coeficientes[m.coeficientes != 0].index)
    # score de cada cliente = soma dos pontos das faixas = fórmula da escala
    fator = 20 / np.log(2)
    woe = m.transformador_.transform(x)
    coef = m.coeficientes
    por_var = (fator * m.logistica_.intercept_[0] + 600 - fator * np.log(50)) / len(coef)
    soma = (fator * woe[coef.index] * coef.values + por_var).sum(axis=1)
    p = m.predict_proba(x)[:, 1]
    esperado = 600 + fator * (np.log(p / (1 - p)) - np.log(50))
    np.testing.assert_allclose(soma, esperado, rtol=1e-8)
    # os pontos da tabela são os mesmos que o cliente recebe (ex.: faixa do -1)
    linha_esp = pontos[(pontos["variavel"] == "VP01") & (pontos["Bin"].astype(str) == "esp_-1")]
    if len(linha_esp):
        cliente = (fator * woe.loc[x["VP01"] == -1, "VP01"].iloc[0] * coef["VP01"] + por_var)
        assert linha_esp["pontos"].iloc[0] == pytest.approx(cliente, abs=0.05)


def test_com_limiar_igual_a_ajustar_de_novo(dados):
    x, y = dados
    base = TransformadorWoE(iv_minimo=0, limiar_correlacao=None, n_jobs=1).fit(x, y)
    assert set(base.variaveis_) == {"VP01", "VP02", "VP03", "estado"}
    direto = TransformadorWoE(iv_minimo=0, limiar_correlacao=0.8, n_jobs=1).fit(x, y)
    copia = base.com_limiar(0.8)
    assert copia.variaveis_ == direto.variaveis_
    assert len(base.variaveis_) == 4  # o original não muda
    pd.testing.assert_frame_equal(copia.transform(x), direto.transform(x))
