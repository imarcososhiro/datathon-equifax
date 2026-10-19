"""Testes de src/equifax/validacao.py com dados inventados.

[IA] Testes gerados com auxílio de IA (Claude). Registrar em docs/uso_ia.md.
"""
import numpy as np
import pandas as pd
import pytest

from equifax.validacao import avaliar, dividir_treino_teste, separar_janelas


@pytest.fixture
def base():
    """Base inventada: 10 safras, ids repetidos entre safras, poucos maus."""
    rng = np.random.default_rng(0)
    n = 4000
    return pd.DataFrame({
        "id": rng.integers(0, 3000, n),
        "safra": rng.choice(range(202301, 202311), n),
        "adimplencia": (rng.random(n) > 0.12).astype(int),
    })


# --- separar_janelas ---------------------------------------------------------

def test_janelas_tem_as_safras_certas(base):
    dev, oot = separar_janelas(base)
    assert set(dev["safra"]) == set(range(202301, 202309))
    assert set(oot["safra"]) == {202309, 202310}
    assert len(dev) + len(oot) == len(base)


def test_janelas_aceitam_safra_como_texto(base):
    base["safra"] = base["safra"].astype(str)
    dev, oot = separar_janelas(base)
    assert len(dev) + len(oot) == len(base)


# --- dividir_treino_teste ----------------------------------------------------

def test_divisao_nao_repete_id_nem_perde_linha(base):
    dev, _ = separar_janelas(base)
    treino, teste = dividir_treino_teste(dev)
    assert not set(treino["id"]) & set(teste["id"])
    assert len(treino) + len(teste) == len(dev)


def test_divisao_tem_proporcao_e_taxa_parecidas(base):
    dev, _ = separar_janelas(base)
    treino, teste = dividir_treino_teste(dev)
    assert len(teste) / len(dev) == pytest.approx(0.25, abs=0.03)
    assert treino["adimplencia"].mean() == pytest.approx(teste["adimplencia"].mean(), abs=0.02)


def test_divisao_e_reprodutivel(base):
    dev, _ = separar_janelas(base)
    _, teste_a = dividir_treino_teste(dev, seed=42)
    _, teste_b = dividir_treino_teste(dev, seed=42)
    assert teste_a.index.equals(teste_b.index)


# --- avaliar -----------------------------------------------------------------

def test_score_perfeito():
    y = np.array([1, 1, 1, 0, 0, 1, 0, 1])
    r = avaliar(y, y.astype(float), np.repeat([202301, 202302], 4))
    total = r.iloc[0]
    assert total["safra"] == "todas"
    assert total["auc"] == 1 and total["gini"] == 1 and total["ks"] == 1


def test_score_invertido_da_auc_zero():
    # Pega erro de convenção: score alto para quem é inadimplente
    y = np.array([1, 1, 1, 0, 0, 1, 0, 1])
    total = avaliar(y, 1.0 - y, np.full(8, 202301)).iloc[0]
    assert total["auc"] == 0 and total["gini"] == -1


def test_score_aleatorio_fica_perto_de_meio():
    rng = np.random.default_rng(1)
    y = (rng.random(20000) > 0.12).astype(int)
    total = avaliar(y, rng.random(20000), np.full(20000, 202301)).iloc[0]
    assert total["auc"] == pytest.approx(0.5, abs=0.02)
    assert total["ks"] < 0.05


def test_uma_linha_por_safra_com_n_e_taxa():
    y = np.array([1, 0, 1, 1, 1, 0, 0, 1])
    safra = np.array([202301] * 4 + [202302] * 4)
    r = avaliar(y, np.linspace(0, 1, 8), safra)
    assert list(r["safra"]) == ["todas", "202301", "202302"]
    assert list(r["n"]) == [8, 4, 4]
    assert r.loc[1, "taxa_inad"] == 0.25 and r.loc[2, "taxa_inad"] == 0.5


def test_safra_com_uma_classe_devolve_nan():
    y = np.array([1, 0, 1, 1])
    r = avaliar(y, np.array([0.9, 0.1, 0.8, 0.7]), np.array([202301, 202301, 202302, 202302]))
    assert np.isnan(r.loc[2, "auc"]) and np.isnan(r.loc[2, "ks"])
    assert not np.isnan(r.loc[1, "auc"])


def test_entradas_invalidas_dao_erro():
    with pytest.raises(ValueError):
        avaliar([1, 0], [0.5], [202301, 202301])            # tamanhos diferentes
    with pytest.raises(ValueError):
        avaliar([1, 0], [0.5, np.nan], [202301, 202301])    # score com NaN
    with pytest.raises(ValueError):
        avaliar([1, 2], [0.5, 0.4], [202301, 202301])       # y fora de 0/1
