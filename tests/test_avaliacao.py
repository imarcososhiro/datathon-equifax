"""Testes de src/equifax/avaliacao.py com scores inventados.

[IA] Testes gerados com auxílio de IA (Claude) em 12/10/2026. Registrar em docs/uso_ia.md.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from equifax.avaliacao import (grafico_calibracao, grafico_distribuicao_score,
                               grafico_ks_por_safra, grafico_roc, psi_score_por_safra,
                               simular_politica, tabela_calibracao, tabela_decis)
from equifax.validacao import avaliar


@pytest.fixture
def dados():
    """Score que ordena bem: a probabilidade de ser bom cresce com o score."""
    rng = np.random.default_rng(0)
    n = 5000
    score = rng.uniform(0.5, 1.0, n)
    y = (rng.random(n) < score).astype(int)
    safra = rng.choice(range(202301, 202309), n)
    return y, score, safra


def test_decis_ordenam_o_risco(dados):
    y, score, _ = dados
    t = tabela_decis(y, score)
    assert len(t) == 10 and t["clientes"].sum() == len(y)
    assert t["clientes"].nunique() <= 2  # faixas do mesmo tamanho
    assert t["taxa_inad"].iloc[0] > t["taxa_inad"].iloc[-1]  # faixa 1 = maior risco
    assert t["score_max"].iloc[0] <= t["score_min"].iloc[1]
    assert t["pct_maus_capturados"].iloc[-1] == pytest.approx(1.0)
    assert t["taxa_inad_acumulada"].iloc[-1] == pytest.approx((y == 0).mean())


def test_decis_com_scores_empatados():
    y = np.array([1, 0] * 50)
    t = tabela_decis(y, np.full(100, 0.7))
    assert (t["clientes"] == 10).all()


def test_psi_do_score(dados):
    y, score, safra = dados
    psi = psi_score_por_safra(score, safra)
    assert list(psi.index) == list(range(202303, 202309))
    assert (psi < 0.05).all()  # mesma distribuição em todas as safras
    deslocado = score.copy()
    deslocado[safra == 202308] -= 0.2
    assert psi_score_por_safra(deslocado, safra)[202308] > 0.25


def test_calibracao(dados):
    y, score, _ = dados
    tabela, brier = tabela_calibracao(y, score)
    # o score inventado é a probabilidade verdadeira: previsto ~ observado
    assert (tabela["inad_prevista"] - tabela["inad_observada"]).abs().max() < 0.05
    assert 0 < brier < 0.25


def test_politica(dados):
    y, score, _ = dados
    p = simular_politica(y, score, taxas_aprovacao=(0.5, 0.9, 1.0))
    assert p.loc[0.5, "aprovados"] == 2500
    assert p.loc[0.5, "inad_aprovados"] < p.loc[0.9, "inad_aprovados"]
    assert p.loc[0.5, "inad_recusados"] > p.loc[0.5, "inad_aprovados"]
    assert p.loc[1.0, "pct_maus_recusados"] == 0
    assert np.isnan(p.loc[1.0, "inad_recusados"])


def test_validacoes():
    with pytest.raises(ValueError, match="NaN"):
        tabela_decis([1, 0], [0.5, np.nan])
    with pytest.raises(ValueError, match="0 e 1"):
        simular_politica([1, 2], [0.5, 0.6])


def test_graficos_nao_quebram(dados):
    y, score, safra = dados
    figs = [grafico_ks_por_safra({"a": avaliar(y, score, safra)}),
            grafico_distribuicao_score(score, safra),
            grafico_roc({"a": (y, score), "b": (y, 1 - score)}),
            grafico_calibracao(y, score)]
    for f in figs:
        assert isinstance(f, plt.Figure)
        plt.close(f)
