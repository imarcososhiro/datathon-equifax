"""Testes de src/equifax/explicar.py com dados inventados.

[IA] Testes gerados com auxílio de IA (Claude) em 13/10/2026. Registrar em docs/uso_ia.md.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from equifax.dados import janelas
from equifax.experimentos import escorar, treinar
from equifax.explicar import (conferir_soma, contribuicoes, escolher_perfis, grafico_beeswarm,
                              grafico_dependencia, grafico_importancia, grafico_shap_por_safra,
                              grafico_waterfall, importancia_shap, shap_por_safra)

VARIAVEIS = ["VP01", "VP02", "VP03", "VP04", "VP05", "faixa_renda", "estado"]


@pytest.fixture
def partes(base):
    return janelas(base)


@pytest.mark.parametrize("modelo,params", [("lgbm", {"n_estimators": 40}), ("xgb", {"n_estimators": 40}),
                                           ("logit_woe", {"n_jobs": 1, "iv_minimo": 0.0})])
def test_contribuicoes_somam_o_score(partes, modelo, params):
    m = treinar(modelo, params, partes["treino"], VARIAVEIS)
    contrib, valor_base, x = contribuicoes(m, partes["teste"], amostra=300)
    assert contrib.shape == (300, len(VARIAVEIS))
    ok, erro = conferir_soma(m, contrib, valor_base, x)
    assert ok, erro


def test_variavel_que_gera_o_target_e_a_mais_importante(partes):
    m = treinar("lgbm", {"n_estimators": 60}, partes["treino"], VARIAVEIS)
    contrib, _, x = contribuicoes(m, partes["teste"])
    imp = importancia_shap(contrib)
    assert set(imp.index[:2]) == {"VP03", "VP04"}  # o target inventado depende só delas
    # VP03 aumenta a chance de ser bom: contribuição cresce com o valor
    assert np.corrcoef(x["VP03"], contrib["VP03"])[0, 1] > 0.5
    assert np.corrcoef(x["VP04"], contrib["VP04"])[0, 1] < -0.5


def test_shap_por_safra_e_perfis(partes):
    m = treinar("lgbm", {"n_estimators": 40}, partes["treino"], VARIAVEIS)
    teste = partes["teste"]
    contrib, _, _ = contribuicoes(m, teste)
    media, media_abs = shap_por_safra(contrib, teste["safra"], ["VP03", "VP04"])
    assert list(media.index) == sorted(teste["safra"].unique())
    assert (media_abs >= 0).all().all()
    score = escorar(m, teste)
    perfis = escolher_perfis(score, indice=teste.index)
    s = dict(zip(teste.index, score))
    assert s[perfis["q05"]] < s[perfis["q50"]] < s[perfis["q95"]]


def test_graficos_nao_quebram(partes):
    m = treinar("lgbm", {"n_estimators": 40}, partes["treino"], VARIAVEIS)
    contrib, valor_base, x = contribuicoes(m, partes["teste"], amostra=300)
    _, media_abs = shap_por_safra(contrib, partes["teste"].loc[contrib.index, "safra"], ["VP03", "VP01"])
    figs = [grafico_importancia(contrib, n=5), grafico_beeswarm(contrib, valor_base, x, n=5),
            grafico_dependencia(contrib, x, "VP01"),   # com o código -1
            grafico_dependencia(contrib, x, "estado"),  # categórica
            grafico_waterfall(contrib, valor_base, x, contrib.index[0]),
            grafico_shap_por_safra(media_abs)]
    for f in figs:
        assert isinstance(f, plt.Figure)
        plt.close("all")
