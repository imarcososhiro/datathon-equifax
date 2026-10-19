"""Testes de src/equifax/experimentos.py com dados inventados.

[IA] Testes gerados com auxílio de IA (Claude) em 11/10/2026. Registrar em docs/uso_ia.md.
"""
import json

import numpy as np
import pandas as pd
import pytest

from equifax.conjuntos import salvar_conjunto
from equifax.dados import janelas
from equifax.experimentos import (COLUNAS_REGISTRO, avaliar_modelo, categorias_do_treino,
                                  escorar, ler_registro, pesos_de_classe, preparar_x, resumir,
                                  rodar_experimento, treinar, treinar_final,
                                  comparar_com_referencia, escolher_vencedor)

VARIAVEIS = ["VP01", "VP02", "VP03", "VP04", "VP05", "faixa_renda", "estado"]
RAPIDO = {"n_estimators": 30}


@pytest.fixture
def partes(base):
    return janelas(base, incluir_oot_interna=True)


@pytest.fixture
def registro(tmp_path):
    return tmp_path / "experimentos.csv"


# --- preparar_x ----------------------------------------------------------------

def test_categorias_vem_so_do_treino(partes):
    treino = partes["treino"].copy()
    treino["estado"] = "Bahia"
    cats = categorias_do_treino(treino, VARIAVEIS)
    assert cats["estado"] == ["Bahia"]
    assert "Sem informação" in cats["faixa_renda"]  # nulo vira categoria
    assert "VP01" not in cats


def test_categoria_nova_vira_ausente_e_numericas_ficam_iguais(partes):
    cats = {"estado": ["São Paulo", "Bahia"], "faixa_renda": ["3 - 1000"]}
    x = preparar_x(partes["teste"], VARIAVEIS, cats)
    assert list(x.columns) == VARIAVEIS
    assert x.loc[partes["teste"]["estado"] == "Paraná", "estado"].isna().all()
    assert (x["VP01"] == -1).any()  # códigos especiais continuam
    pd.testing.assert_series_equal(x["VP03"], partes["teste"]["VP03"])


def test_variavel_ausente_da_erro(partes):
    with pytest.raises(KeyError, match="VP99"):
        preparar_x(partes["teste"], ["VP01", "VP99"], {})


# --- treinar / avaliar -----------------------------------------------------------

@pytest.mark.parametrize("modelo,params", [("lgbm", RAPIDO), ("xgb", RAPIDO),
                                           ("logit_woe", {"n_jobs": 1})])
def test_treinar_e_avaliar(partes, modelo, params):
    treinado = treinar(modelo, params, partes["treino"], VARIAVEIS, semente=1)
    metricas, imp = avaliar_modelo(treinado, partes["teste"])
    assert metricas.loc[0, "safra"] == "todas"
    assert metricas.loc[0, "auc"] > 0.6  # o target depende de VP03 e VP04
    assert imp.sum() == pytest.approx(1.0)
    assert list(imp.index) == VARIAVEIS
    score = escorar(treinado, partes["teste"])
    assert ((score >= 0) & (score <= 1)).all() and not np.isnan(score).any()


def test_modelo_desconhecido_da_erro(partes):
    with pytest.raises(KeyError, match="Disponíveis"):
        treinar("rede_neural", None, partes["treino"], VARIAVEIS)


def test_params_completam_a_referencia(partes):
    treinado = treinar("lgbm", RAPIDO, partes["treino"], VARIAVEIS)
    assert treinado.params["n_estimators"] == 30
    assert treinado.params["num_leaves"] == 31  # veio da referência


# --- rodar_experimento -----------------------------------------------------------

def test_registro_grava_o_que_foi_rodado(partes, registro):
    res = rodar_experimento("exp_a", VARIAVEIS, partes, params=RAPIDO, sementes=(1, 2),
                            caminho_registro=registro, nome_conjunto="inventado")
    lido = ler_registro(registro)
    assert list(lido.columns) == COLUNAS_REGISTRO
    assert len(lido) == 2
    assert list(lido["semente"]) == [1, 2]
    assert (lido["nome"] == "exp_a").all() and (lido["conjunto"] == "inventado").all()
    assert (lido["n_variaveis"] == len(VARIAVEIS)).all()
    assert (lido["avaliado_em"] == "teste").all()
    assert json.loads(lido.loc[0, "params"])["n_estimators"] == 30
    assert lido["auc"].tolist() == pytest.approx(res.registro["auc"].tolist())
    assert (lido["ks_min_safra"] <= lido["ks"] + 0.5).all()
    assert lido["safra_ks_min"].str.match(r"2023\d\d").all()
    assert set(res.por_safra["semente"]) == {1, 2}

    rodar_experimento("exp_b", VARIAVEIS, partes, params=RAPIDO, sementes=(1,),
                      caminho_registro=registro)
    assert len(ler_registro(registro)) == 3  # acrescenta, não apaga
    tabela = resumir(ler_registro(registro))
    assert len(tabela) == 2


def test_mesma_semente_mesma_metrica(partes, registro):
    a = rodar_experimento("a", VARIAVEIS, partes, params=RAPIDO, sementes=(7,), registrar=False)
    b = rodar_experimento("b", VARIAVEIS, partes, params=RAPIDO, sementes=(7,), registrar=False)
    assert a.registro.loc[0, "auc"] == b.registro.loc[0, "auc"]
    assert a.registro.loc[0, "ks"] == b.registro.loc[0, "ks"]
    assert not registro.exists()


def test_oot_interna_exige_parametro_explicito(partes, registro):
    with pytest.raises(ValueError, match="usar_oot_interna=True"):
        rodar_experimento("x", VARIAVEIS, partes, avaliar_em="oot_interna",
                          caminho_registro=registro)
    assert not registro.exists()
    res = rodar_experimento("x", VARIAVEIS, partes, params=RAPIDO, sementes=(1,),
                            avaliar_em="oot_interna", usar_oot_interna=True,
                            caminho_registro=registro)
    assert res.registro.loc[0, "avaliado_em"] == "oot_interna"


def test_oot_interna_ausente_das_partes_da_erro(base, registro):
    partes = janelas(base)  # sem a OOT interna
    with pytest.raises(KeyError):
        rodar_experimento("x", VARIAVEIS, partes, avaliar_em="oot_interna",
                          usar_oot_interna=True, caminho_registro=registro)


def test_nao_avalia_no_treino(partes, registro):
    with pytest.raises(ValueError, match="treino"):
        rodar_experimento("x", VARIAVEIS, partes, avaliar_em="treino", caminho_registro=registro)


def test_conjunto_pelo_nome(partes, registro, tmp_path):
    caminho_conjuntos = tmp_path / "conjuntos.yaml"
    salvar_conjunto("pequeno", ["VP03", "VP04"], caminho=caminho_conjuntos)
    res = rodar_experimento("x", "pequeno", partes, params=RAPIDO, sementes=(1,),
                            caminho_registro=registro, caminho_conjuntos=caminho_conjuntos)
    assert res.registro.loc[0, "conjunto"] == "pequeno"
    assert res.registro.loc[0, "n_variaveis"] == 2
    assert list(res.importancia.index) == ["VP03", "VP04"]


def test_registro_com_outras_colunas_da_erro(partes, registro):
    registro.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outras colunas"):
        rodar_experimento("x", VARIAVEIS, partes, params=RAPIDO, sementes=(1,),
                          caminho_registro=registro)


# --- peso de classe e treino final ---------------------------------------------------

def test_pesos_de_classe_equilibram():
    y = np.array([1] * 90 + [0] * 10)
    p = pesos_de_classe(y)
    assert p.sum() == pytest.approx(100)
    assert p[y == 0].sum() == pytest.approx(p[y == 1].sum())


@pytest.mark.parametrize("modelo", ["lgbm", "xgb"])
def test_balancear_muda_o_modelo_e_fica_registrado(partes, modelo):
    sem = treinar(modelo, RAPIDO, partes["treino"], VARIAVEIS, semente=1)
    com = treinar(modelo, {**RAPIDO, "balancear": True}, partes["treino"], VARIAVEIS, semente=1)
    assert com.params["balancear"] is True
    # com mais peso nos maus, a probabilidade média de ser bom cai
    assert escorar(com, partes["teste"]).mean() < escorar(sem, partes["teste"]).mean() - 0.05


def test_treinar_final_usa_a_janela(base):
    t8 = treinar_final("lgbm", RAPIDO, base, VARIAVEIS, janela=("202301", "202308"))
    t10 = treinar_final("lgbm", RAPIDO, base, VARIAVEIS, janela=("202301", "202310"))
    n8 = base["safra"].between(202301, 202308).sum()
    assert t8.estimador.n_features_in_ == len(VARIAVEIS)
    assert t8.estimador._n_classes == 2 and n8 < len(base)
    with pytest.raises(ValueError, match="Nenhuma linha"):
        treinar_final("lgbm", RAPIDO, base, VARIAVEIS, janela=("202401", "202402"))


def test_comparar_com_referencia(partes, registro):
    rodar_experimento("ref", ["VP03"], partes, params=RAPIDO, sementes=(1, 2), caminho_registro=registro)
    rodar_experimento("mais", ["VP03", "VP04"], partes, params=RAPIDO, sementes=(1, 2),
                      caminho_registro=registro)
    t = comparar_com_referencia(ler_registro(registro), "ref", ["mais"])
    assert list(t.index) == ["ref", "mais"]
    assert t.loc["ref", "delta_auc"] == 0
    assert t.loc["mais", "delta_auc"] > 0.02 and t.loc["mais", "supera_ruido"]


def test_comparar_usa_so_a_ultima_rodada(partes, registro):
    for _ in range(2):  # o mesmo notebook rodado duas vezes
        rodar_experimento("ref", ["VP03"], partes, params=RAPIDO, sementes=(1, 2),
                          caminho_registro=registro)
        rodar_experimento("mais", ["VP03", "VP04"], partes, params=RAPIDO, sementes=(1, 2),
                          caminho_registro=registro)
    duas_rodadas = ler_registro(registro)
    assert len(duas_rodadas) == 8
    t = comparar_com_referencia(duas_rodadas, "ref", ["mais"])
    uma_rodada = comparar_com_referencia(duas_rodadas.iloc[4:], "ref", ["mais"])
    pd.testing.assert_frame_equal(t, uma_rodada)
    pd.testing.assert_frame_equal(resumir(duas_rodadas), resumir(duas_rodadas.iloc[4:]))


def test_comparar_sem_ruido_nao_decide(partes, registro):
    rodar_experimento("ref1", ["VP03"], partes, params=RAPIDO, sementes=(1,), caminho_registro=registro)
    rodar_experimento("mais1", ["VP03", "VP04"], partes, params=RAPIDO, sementes=(1,),
                      caminho_registro=registro)
    t = comparar_com_referencia(ler_registro(registro), "ref1", ["mais1"])
    assert t.loc["mais1", "supera_ruido"] is None  # uma semente só: sem critério


# --- escolher_vencedor (critério D7) ----------------------------------------------
# [IA] Testes gerados com auxílio de IA (Claude) em 18/10/2026.

def _comparacao(linhas):
    colunas = ["auc_teste", "auc_dp_teste", "ks_min_safra_teste", "ks_min_safra_dp_teste",
               "auc_temporal", "auc_dp_temporal"]
    return pd.DataFrame(linhas, index=["M1", "M2", "M3"][:len(linhas)], columns=colunas)


def test_empate_fica_o_mais_simples():
    comp = _comparacao([[0.7500, 0.0002, 0.350, 0.002, 0.7431, 0.0002],
                        [0.7501, 0.0002, 0.346, 0.002, 0.7432, 0.0002],
                        [0.7502, 0.0003, 0.347, 0.002, 0.7432, 0.0003]])
    vencedor, motivo = escolher_vencedor(comp)
    assert vencedor == "M1"
    assert not motivo.loc["M2", "fica"] and not motivo.loc["M3", "fica"]


def test_mais_complexo_vence_se_ganha_nos_dois_protocolos():
    comp = _comparacao([[0.7500, 0.0002, 0.350, 0.002, 0.7431, 0.0002],
                        [0.7501, 0.0002, 0.350, 0.002, 0.7432, 0.0002],
                        [0.7527, 0.0003, 0.351, 0.002, 0.7467, 0.0003]])
    vencedor, motivo = escolher_vencedor(comp)
    assert vencedor == "M3"
    assert motivo.loc["M3", "contra"] == "M1"


def test_ganho_so_no_teste_nao_basta():
    # o ganho no teste por cliente pode vir de reconhecer o mês (macro)
    comp = _comparacao([[0.7500, 0.0002, 0.350, 0.002, 0.7431, 0.0002],
                        [0.7507, 0.0002, 0.350, 0.002, 0.7431, 0.0002]])
    vencedor, motivo = escolher_vencedor(comp)
    assert vencedor == "M1"
    assert "validação temporal" in motivo.loc["M2", "motivo"]


def test_perda_de_estabilidade_impede_a_troca():
    comp = _comparacao([[0.7500, 0.0002, 0.350, 0.002, 0.7431, 0.0002],
                        [0.7517, 0.0002, 0.333, 0.002, 0.7447, 0.0002]])
    vencedor, motivo = escolher_vencedor(comp)
    assert vencedor == "M1"
    assert "pior KS" in motivo.loc["M2", "motivo"]


def test_m3_e_comparado_com_o_vencedor_atual():
    # M2 vence o M1; o M3 precisa ganhar do M2, não do M1
    comp = _comparacao([[0.7600, 0.0002, 0.380, 0.002, 0.7550, 0.0002],
                        [0.7650, 0.0002, 0.385, 0.002, 0.7600, 0.0002],
                        [0.7651, 0.0002, 0.385, 0.002, 0.7601, 0.0002]])
    vencedor, motivo = escolher_vencedor(comp)
    assert vencedor == "M2"
    assert motivo.loc["M3", "contra"] == "M2"


def test_faltando_coluna_da_erro():
    with pytest.raises(ValueError):
        escolher_vencedor(pd.DataFrame({"auc_teste": [0.7]}, index=["M1"]))
