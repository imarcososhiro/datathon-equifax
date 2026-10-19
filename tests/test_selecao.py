"""Testes de src/equifax/selecao.py com dados inventados.

[IA] Testes gerados com auxílio de IA (Claude). Registrar em docs/uso_ia.md.
"""
import numpy as np
import pandas as pd
import pytest

from equifax.selecao import (ajustar_binning, classificar_iv, classificar_psi,
                             correlacao_spearman, ficha_variaveis, iv_de, pares_correlacionados,
                             psi, psi_por_safra, selecionar_por_correlacao, tabela_woe,
                             woe_por_categoria)


@pytest.fixture(scope="module")
def treino():
    """Base inventada com um caso de cada tipo que a ficha precisa tratar."""
    rng = np.random.default_rng(0)
    n = 8000
    risco = rng.normal(size=n)
    bom = (rng.random(n) > 1 / (1 + np.exp(-(-2.0 + 1.0 * risco)))).astype(int)
    safra = rng.choice(range(202301, 202309), n)
    df = pd.DataFrame({"safra": safra, "adimplencia": bom})
    df["preditiva"] = -risco + rng.normal(scale=2.0, size=n)
    df["ruido"] = rng.normal(size=n)
    df["constante"] = 5.0
    df["vazamento"] = np.where(bom == 1, rng.normal(3, 1, n), rng.normal(-3, 1, n))
    df["instavel"] = rng.normal(size=n) + (safra - 202301) * 0.4
    codigos = rng.normal(size=n) * 10
    codigos[rng.random(n) < 0.5] = -1
    codigos[rng.random(n) < 0.1] = -2
    df["com_codigos"] = codigos
    faixa = pd.cut(risco, [-9, -0.5, 0.5, 9], labels=["baixa", "media", "alta"]).astype(object)
    faixa[rng.random(n) < 0.05] = None
    df["faixa"] = faixa
    return df


# --- WoE e IV ----------------------------------------------------------------

def test_iv_e_woe_batem_com_a_conta_manual():
    # 2 valores, sem agrupamento possível: dá para conferir na mão
    x = pd.Series([0.0] * 500 + [1.0] * 500, name="x")
    bom = pd.Series([1] * 450 + [0] * 50 + [1] * 350 + [0] * 150)
    ob = ajustar_binning(x, bom, min_prebin_size=0.05)
    pct_bons = np.array([450, 350]) / 800
    pct_maus = np.array([50, 150]) / 200
    woe = np.log(pct_bons / pct_maus)
    assert iv_de(ob) == pytest.approx(((pct_bons - pct_maus) * woe).sum(), rel=1e-6)
    tabela = tabela_woe(ob)
    assert tabela["WoE"].astype(float).tolist() == pytest.approx(woe.tolist(), rel=1e-6)
    assert tabela["WoE"].iloc[0] > 0  # faixa com mais bons tem WoE positivo (Siddiqi)


def test_cada_codigo_especial_fica_em_faixa_propria(treino):
    ob = ajustar_binning(treino["com_codigos"], treino["adimplencia"])
    tabela = tabela_woe(ob).set_index("Bin")
    assert tabela.loc["esp_-1", "total"] == (treino["com_codigos"] == -1).sum()
    assert tabela.loc["esp_-2", "total"] == (treino["com_codigos"] == -2).sum()
    assert tabela["total"].sum() == len(treino)


def test_woe_por_categoria_bate_com_a_conta_manual():
    x = pd.Series(["a"] * 500 + ["b"] * 400 + [None] * 100)
    bom = pd.Series([1] * 450 + [0] * 50 + [1] * 300 + [0] * 100 + [1] * 50 + [0] * 50)
    tabela = woe_por_categoria(x, bom)
    assert list(tabela.index) == ["Sem informação", "a", "b"]  # o nulo vira categoria
    assert tabela["total"].tolist() == [100, 500, 400]
    pct_bons = np.array([50, 450, 300]) / 800
    pct_maus = np.array([50, 50, 100]) / 200
    woe = np.log(pct_bons / pct_maus)
    assert tabela["WoE"].tolist() == pytest.approx(woe.tolist())
    assert tabela["IV"].sum() == pytest.approx(((pct_bons - pct_maus) * woe).sum())
    assert tabela.loc["a", "taxa_inadimplencia"] == pytest.approx(0.1)


def test_classificar_iv():
    assert [classificar_iv(v) for v in (0.01, 0.05, 0.2, 0.4, 0.8)] == [
        "sem poder", "fraco", "médio", "forte", "suspeito (revisar)"]


# --- PSI ---------------------------------------------------------------------

def test_psi_de_distribuicoes_iguais_e_zero():
    x = pd.Series(np.random.default_rng(1).normal(size=5000))
    assert psi(x, x) == pytest.approx(0)


def test_psi_bate_com_a_conta_manual():
    ref = pd.Series([1.0] * 50 + [2.0] * 50)
    atual = pd.Series([1.0] * 80 + [2.0] * 20)
    esperado = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert psi(ref, atual) == pytest.approx(esperado)


def test_psi_conta_valores_fora_do_intervalo_da_referencia():
    rng = np.random.default_rng(2)
    ref = pd.Series(rng.normal(size=5000))
    atual = pd.Series(rng.normal(size=5000) + 10)  # tudo acima do máximo da referência
    assert psi(ref, atual) > 1


def test_psi_enxerga_mudanca_nos_codigos_e_nos_nulos():
    rng = np.random.default_rng(3)
    ref = pd.Series(rng.normal(size=5000))
    com_codigo = ref.copy()
    com_codigo[:1500] = -1
    com_nulo = ref.copy()
    com_nulo[:1500] = np.nan
    assert psi(ref, com_codigo) > 0.25
    assert psi(ref, com_nulo) > 0.25


def test_psi_aceita_variavel_de_texto_com_muitas_categorias():
    rng = np.random.default_rng(4)
    categorias = [f"c{i}" for i in range(30)]
    ref = pd.Series(rng.choice(categorias, 6000))
    assert psi(ref, pd.Series(rng.choice(categorias, 6000))) < 0.05
    assert psi(ref, pd.Series(rng.choice(categorias[:10], 6000))) > 0.25


def test_psi_por_safra_usa_as_duas_primeiras_como_referencia(treino):
    psis = psi_por_safra(treino, "instavel")
    assert list(psis.index) == list(range(202303, 202309))
    assert psis.is_monotonic_increasing  # o desvio cresce a cada safra
    assert psi_por_safra(treino, "ruido").max() < 0.10


def test_classificar_psi():
    assert [classificar_psi(v) for v in (0.05, 0.15, 0.3, np.nan)] == [
        "estável", "atenção", "instável", "sem dado"]


# --- Ficha -------------------------------------------------------------------

@pytest.fixture(scope="module")
def ficha_e_binnings(treino):
    variaveis = ["preditiva", "ruido", "constante", "vazamento", "instavel", "com_codigos", "faixa"]
    return ficha_variaveis(treino, variaveis, categoricas=("faixa",), verbose=False)


def test_ficha_classifica_cada_caso(ficha_e_binnings):
    ficha, _ = ficha_e_binnings
    assert ficha.loc["constante", "sugestao"] == "remover (constante)"
    assert ficha.loc["vazamento", "sugestao"] == "revisar vazamento"
    assert ficha.loc["instavel", "sugestao"] == "investigar estabilidade"
    assert ficha.loc["ruido", "sugestao"] == "candidata a remoção (testar)"
    assert ficha.loc["preditiva", "sugestao"] == "manter"
    assert ficha.loc["preditiva", "iv"] > ficha.loc["ruido", "iv"]
    assert ficha["iv"].is_monotonic_decreasing  # ordenada do maior IV para o menor


def test_ficha_registra_codigos_e_nulos(ficha_e_binnings, treino):
    ficha, binnings = ficha_e_binnings
    assert ficha.loc["com_codigos", "pct_-1"] == pytest.approx((treino["com_codigos"] == -1).mean())
    assert ficha.loc["faixa", "pct_nan"] == pytest.approx(treino["faixa"].isna().mean())
    # o nulo da categórica entra como "Sem informação", sem perder linha
    assert tabela_woe(binnings["faixa"])["total"].sum() == len(treino)
    assert "constante" not in binnings


def test_ficha_nao_altera_o_treino(treino):
    antes = treino.copy()
    ficha_variaveis(treino, ["ruido", "faixa"], categoricas=("faixa",), verbose=False)
    pd.testing.assert_frame_equal(treino, antes)


# --------------------------------------------------------------------------- #
# Correlação (11/10/2026)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def base_correlacao():
    """a e a_escala são a mesma informação; b é parecida com a; c é independente."""
    rng = np.random.default_rng(1)
    n = 3000
    a = rng.normal(size=n)
    df = pd.DataFrame({
        "a": a,
        "a_escala": np.exp(a),                     # mesma ordem de a
        "b": a + rng.normal(scale=0.3, size=n),
        "c": rng.normal(size=n),
    })
    df.loc[:9, "a"] = 9999                         # código especial no topo da escala
    return df


def test_spearman_ignora_a_escala_e_o_codigo_especial(base_correlacao):
    corr = correlacao_spearman(base_correlacao, ["a", "a_escala", "b", "c"])
    assert corr.loc["a", "a_escala"] > 0.99
    assert abs(corr.loc["a", "c"]) < 0.1
    assert np.allclose(np.diag(corr), 1)


def test_pares_correlacionados_respeita_o_limiar(base_correlacao):
    corr = correlacao_spearman(base_correlacao, ["a", "a_escala", "b", "c"])
    pares = pares_correlacionados(corr, limiar=0.90)
    nomes = {frozenset(p) for p in zip(pares["var_1"], pares["var_2"])}
    assert frozenset({"a", "a_escala"}) in nomes
    assert not any("c" in p for p in nomes)
    assert pares["correlacao"].abs().is_monotonic_decreasing


def test_selecao_por_correlacao_mantem_a_de_maior_iv(base_correlacao):
    corr = correlacao_spearman(base_correlacao, ["a", "a_escala", "b", "c"])
    iv = pd.Series({"a": 0.10, "a_escala": 0.20, "b": 0.05, "c": 0.01})
    ficam, saem = selecionar_por_correlacao(corr, iv, limiar=0.90)
    assert ficam == ["a_escala", "c"]
    assert saem == {"a": "a_escala", "b": "a_escala"}
