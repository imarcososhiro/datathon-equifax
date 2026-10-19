"""Testes de src/equifax/ajuste.py com dados inventados (buscas bem curtas).

[IA] Testes gerados com auxílio de IA (Claude) em 12/10/2026. Registrar em docs/uso_ia.md.
"""
import json

import numpy as np
import pytest

from equifax.ajuste import (ajustar_boosting, ajustar_logistica_woe, avaliar_cv,
                            comparar_balanceamento, dobras_por_id)
from equifax.dados import janelas
from equifax.experimentos import escorar, treinar

VARIAVEIS = ["VP01", "VP02", "VP03", "VP04", "VP05", "faixa_renda", "estado"]


@pytest.fixture
def treino(base):
    return janelas(base)["treino"]


def test_dobras_agrupadas_e_cobrem_o_treino(treino):
    dobras = dobras_por_id(treino, n_dobras=4)
    assert len(dobras) == 4
    validacao = np.concatenate([b for _, b in dobras])
    assert sorted(validacao) == list(range(len(treino)))  # cada linha valida uma vez
    ids = treino["id"].to_numpy()
    for a, b in dobras:
        assert not set(ids[a]) & set(ids[b])


def test_avaliar_cv_com_parada_antecipada(treino):
    dobras = dobras_por_id(treino, 3)
    r = avaliar_cv("lgbm", {"learning_rate": 0.1}, treino, VARIAVEIS, dobras, parada_antecipada=10)
    assert 0.6 < r["auc_cv"] < 1 and r["auc_cv_dp"] >= 0
    assert 1 <= r["n_estimators"] < 3000
    r2 = avaliar_cv("lgbm", {"learning_rate": 0.1}, treino, VARIAVEIS, dobras, parada_antecipada=10)
    assert r == {**r2, "tempo_s": r["tempo_s"]}  # reprodutível


@pytest.mark.parametrize("modelo", ["lgbm", "xgb"])
def test_ajustar_boosting_devolve_params_prontos(treino, modelo):
    params, tabela = ajustar_boosting(modelo, treino, VARIAVEIS, n_tentativas=3, n_dobras=3,
                                      parada_antecipada=10, mostrar=False)
    assert len(tabela) == 3
    assert tabela["auc_cv"].is_monotonic_decreasing
    assert params["n_estimators"] == tabela.loc[0, "n_estimators"]
    assert isinstance(params["balancear"], bool)
    json.dumps(params)  # vai para o registro sem erro de tipo
    treinado = treinar(modelo, params, treino, VARIAVEIS)
    assert escorar(treinado, treino).shape == (len(treino),)


def test_mesma_semente_mesma_busca(treino):
    a, _ = ajustar_boosting("lgbm", treino, VARIAVEIS, n_tentativas=2, n_dobras=3,
                            parada_antecipada=10, mostrar=False)
    b, _ = ajustar_boosting("lgbm", treino, VARIAVEIS, n_tentativas=2, n_dobras=3,
                            parada_antecipada=10, mostrar=False)
    assert a == b


def test_comparar_balanceamento(treino):
    t = comparar_balanceamento("lgbm", {"learning_rate": 0.1}, treino, VARIAVEIS, n_dobras=3,
                               parada_antecipada=10)
    assert list(t.index) == [False, True]
    assert t["auc_cv"].between(0.6, 1).all()


def test_ajustar_logistica_woe(treino):
    params, tabela = ajustar_logistica_woe(treino, VARIAVEIS, valores_C=(0.01, 1.0),
                                           l1_ratios=(0.0, 1.0), limiares=(0.7, 0.9),
                                           n_dobras=3, n_jobs=1, mostrar=False)
    assert len(tabela) == 2 * 2 * 2
    assert tabela["convergiu"].all()
    assert set(params) == {"C", "l1_ratio", "iv_minimo", "limiar_correlacao"}
    treinado = treinar("logit_woe", {**params, "n_jobs": 1}, treino, VARIAVEIS)
    assert treinado.estimador.convergiu_
