"""Testes de src/equifax/features.py com dados inventados.

[IA] Testes gerados com auxílio de IA (Claude) em 13/10/2026. Registrar em docs/uso_ia.md.
"""
import numpy as np
import pandas as pd
import pytest

from equifax.dados import janelas
from equifax.experimentos import rodar_experimento
from equifax.features import (ESTADO_PARA_REGIAO, REGIOES, Cep, ContagensPorLinha, Interacoes,
                              RendaRelativa, ajustar_familias, gerar_features, regiao_do_estado,
                              uf_do_cep)
from equifax.tratamento import ESTADOS, SEM_INFORMACAO


def test_contagens_so_nas_vps_da_lista():
    df = pd.DataFrame({"VP01": [-1, 0, np.nan], "VP02": [-1, 9999, -2], "VP03": [-1, -1, -1]})
    f = ContagensPorLinha(["VP01", "VP02"]).aplicar(df)
    assert list(f["fe_n_menos1"]) == [2, 0, 0]
    assert list(f["fe_n_9999"]) == [0, 1, 0]
    assert list(f["fe_n_menos2"]) == [0, 0, 1]
    assert list(f["fe_n_zero"]) == [0, 1, 0]
    assert list(f["fe_n_nulo"]) == [0, 0, 1]


def test_regioes_cobrem_os_26_estados_e_o_df():
    estados = [e for regiao, lista in REGIOES.items() if regiao != SEM_INFORMACAO for e in lista]
    assert sorted(estados) == sorted(ESTADOS)
    assert ESTADO_PARA_REGIAO["Distrito Federal"] == "Centro-Oeste"


def test_regiao_do_estado_desconhecido_vira_sem_informacao():
    estados = pd.Series(["São Paulo", "Sao Paulo", np.nan])
    assert list(regiao_do_estado(estados)) == ["Sudeste", SEM_INFORMACAO, SEM_INFORMACAO]


def test_renda_relativa_aprende_so_no_treino():
    treino = pd.DataFrame({"faixa_renda": ["3 - 1000", "1001 - 2000", "2001 - 3000",
                                           "5001 - 7000", "7001 - 10000"],
                           "estado": ["Bahia", "Bahia", "Bahia", "São Paulo", "São Paulo"]})
    fam = RendaRelativa().ajustar(treino)
    teste = pd.DataFrame({"faixa_renda": ["2001 - 3000", "2001 - 3000", "2001 - 3000"],
                          "estado": ["Bahia", "São Paulo", "Acre"]})
    f = fam.aplicar(teste)
    assert list(f["fe_renda_ord"]) == [2, 2, 2]
    assert f.loc[0, "fe_renda_menos_mediana_uf"] == 1      # mediana da Bahia = 1
    assert f.loc[1, "fe_renda_menos_mediana_uf"] == -2.5   # mediana de SP = 4,5
    assert f.loc[2, "fe_renda_menos_mediana_uf"] == 0      # Acre não visto: mediana geral (2)
    assert f.loc[0, "fe_renda_pct_uf"] == pytest.approx(1.0)  # todo o treino da Bahia <= 2
    assert f.loc[0, "fe_renda_menos_mediana_regiao"] == 1  # Nordeste = só a Bahia no treino


def test_interacoes_agrupam_combinacoes_raras():
    treino = pd.DataFrame({"faixa_renda": ["3 - 1000"] * 5 + ["7001 - 10000"],
                           "faixa_idade": ["18 - 25"] * 5 + ["76 - 99"],
                           "estado": ["Bahia"] * 6})
    fam = Interacoes(minimo=3).ajustar(treino)
    f = fam.aplicar(treino)
    assert f.loc[0, "fe_renda_x_idade"] == "3 - 1000 | 18 - 25"
    assert f.loc[5, "fe_renda_x_idade"] == "outros"
    assert f.loc[0, "fe_idade_x_regiao"] == "18 - 25 | Nordeste"
    assert list(f["fe_renda_x_idade"].cat.categories) == ["3 - 1000 | 18 - 25", "outros"]
    assert f.loc[0, "fe_idade_ord"] == 0 and f.loc[5, "fe_idade_ord"] == 1


def test_uf_do_cep():
    cep = pd.Series(["01310100", "20040002", "69301000", "70040010", "Sem informação", "123"])
    assert list(uf_do_cep(cep).fillna("-")) == ["São Paulo", "Rio de Janeiro", "Roraima",
                                                 "Distrito Federal", "-", "-"]


def test_cep_indicadores():
    base = pd.DataFrame({"cep8": ["01310100", "01310000", "Sem informação", "20040002"],
                         "estado": ["São Paulo", "Bahia", "Bahia", "Sem informação"],
                         "adimplencia": [1, 0, 1, 1], "id": [1, 2, 3, 4]})
    f = Cep().ajustar(base).aplicar(base)
    assert list(f["fe_cep_ausente"]) == [0, 0, 1, 0]
    assert list(f["fe_cep_generico"]) == [0, 1, 0, 0]
    assert f.loc[0, "fe_cep_coerente_uf"] == 1 and f.loc[1, "fe_cep_coerente_uf"] == 0
    assert np.isnan(f.loc[2, "fe_cep_coerente_uf"]) and np.isnan(f.loc[3, "fe_cep_coerente_uf"])


def test_taxa_do_cep_no_treino_vem_de_fora_da_dobra(base):
    treino = janelas(base)["treino"].copy()
    # um prefixo que só aparece numa linha, de um mau: dentro da amostra a taxa seria 100%
    treino.iloc[:1, treino.columns.get_loc("cep8")] = "99900000"
    treino.iloc[:1, treino.columns.get_loc("adimplencia")] = 0
    fam = Cep(suavizacao=0)
    dentro = fam.ajustar(treino).aplicar(treino)["fe_cep3_taxa_inad"]
    fora = fam.ajustar_aplicar(treino)["fe_cep3_taxa_inad"]
    assert dentro.iloc[0] == pytest.approx(1.0)  # vê a própria resposta
    assert fora.iloc[0] < 1.0                    # não vê


def test_gerar_features_nao_altera_as_partes(base):
    partes = janelas(base)
    vps = [c for c in base.columns if c.startswith("VP")]
    novas, colunas = gerar_features(partes, [ContagensPorLinha(vps), RendaRelativa(),
                                             Interacoes(minimo=50), Cep()])
    assert set(colunas) == {"f1_contagens", "f2_renda_relativa", "f3_interacoes", "f4_cep"}
    for k in partes:
        assert len(novas[k]) == len(partes[k]) and novas[k].index.equals(partes[k].index)
        assert not any(c.startswith("fe_") for c in partes[k].columns)
    todas = sum(colunas.values(), [])
    assert all(c in novas["teste"].columns for c in todas)


def test_features_entram_no_motor(base, tmp_path):
    partes = janelas(base)
    novas, colunas = gerar_features(partes, [Interacoes(minimo=50), Cep()])
    variaveis = ["VP03", "VP04", "faixa_renda"] + colunas["f3_interacoes"] + colunas["f4_cep"]
    res = rodar_experimento("x", variaveis, novas, params={"n_estimators": 30}, sementes=(1,),
                            caminho_registro=tmp_path / "r.csv")
    assert res.registro.loc[0, "auc"] > 0.6


def test_ajustar_familias_aplica_igual_ao_gerar_features(base):
    partes = janelas(base)
    esperado, _ = gerar_features(partes, [Cep(suavizacao=50), RendaRelativa()])
    treino_fe, aplicar = ajustar_familias(partes["treino"], [Cep(suavizacao=50), RendaRelativa()])
    pd.testing.assert_frame_equal(treino_fe, esperado["treino"])
    pd.testing.assert_frame_equal(aplicar(partes["teste"]), esperado["teste"])
