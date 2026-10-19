"""Testes de src/equifax/conjuntos.py com listas e fichas inventadas.

[IA] Testes gerados com auxílio de IA (Claude) em 11/10/2026. Registrar em docs/uso_ia.md.
"""
import numpy as np
import pandas as pd
import pytest
import yaml

from equifax.conjuntos import (carregar_conjunto, derivar_conjunto, gerar_conjuntos_do_funil,
                               listar_conjuntos, salvar_conjunto, salvar_conjuntos_do_funil)


@pytest.fixture
def caminho(tmp_path):
    return tmp_path / "config" / "conjuntos.yaml"


def test_salvar_e_carregar_mantem_a_ordem(caminho):
    salvar_conjunto("teste", ["VP10", "VP02", "faixa_renda"], "um teste", caminho=caminho)
    assert carregar_conjunto("teste", caminho) == ["VP10", "VP02", "faixa_renda"]
    tabela = listar_conjuntos(caminho)
    assert tabela.loc[0, "conjunto"] == "teste"
    assert tabela.loc[0, "n_variaveis"] == 3


def test_conjunto_inexistente_da_erro(caminho):
    salvar_conjunto("a", ["VP01"], caminho=caminho)
    with pytest.raises(KeyError, match="Disponíveis"):
        carregar_conjunto("b", caminho)


def test_listar_sem_arquivo_devolve_tabela_vazia(caminho):
    assert listar_conjuntos(caminho).empty


def test_nao_aceita_repetidas_nem_vazio_nem_nome_ruim(caminho):
    with pytest.raises(ValueError, match="repetidas"):
        salvar_conjunto("a", ["VP01", "VP01"], caminho=caminho)
    with pytest.raises(ValueError, match="vazio"):
        salvar_conjunto("a", [], caminho=caminho)
    with pytest.raises(ValueError, match="nome"):
        salvar_conjunto("com espaço", ["VP01"], caminho=caminho)


def test_nao_sobrescreve_sem_pedir(caminho):
    salvar_conjunto("a", ["VP01", "VP02"], caminho=caminho)
    salvar_conjunto("a", ["VP01", "VP02"], caminho=caminho)  # igual: não dá erro
    with pytest.raises(ValueError, match="sobrescrever"):
        salvar_conjunto("a", ["VP01"], caminho=caminho)
    salvar_conjunto("a", ["VP01"], caminho=caminho, sobrescrever=True)
    assert carregar_conjunto("a", caminho) == ["VP01"]


def test_varios_conjuntos_no_mesmo_arquivo(caminho):
    salvar_conjunto("a", ["VP01"], caminho=caminho)
    salvar_conjunto("b", ["VP02", "VP03"], caminho=caminho)
    assert list(listar_conjuntos(caminho)["conjunto"]) == ["a", "b"]
    conteudo = yaml.safe_load(caminho.read_text(encoding="utf-8"))
    assert set(conteudo["conjuntos"]) == {"a", "b"}


# --- conjuntos do funil -------------------------------------------------------

@pytest.fixture
def ficha_inventada():
    """10 VPs e 3 categóricas: 1 constante, 1 instável, 2 saem por correlação."""
    variaveis = [f"VP{i:02d}" for i in range(1, 11)] + ["faixa_renda", "faixa_idade", "estado"]
    ficha = pd.DataFrame(index=pd.Index(variaveis, name="variavel"))
    ficha["sugestao"] = "manter"
    ficha["classe_psi"] = "estável"
    ficha.loc["VP03", "sugestao"] = "remover (constante)"
    ficha.loc["VP05", ["sugestao", "classe_psi"]] = ["investigar estabilidade", "instável"]
    ficha.loc[["VP07", "VP08", "VP09"], "sugestao"] = "candidata a remoção (testar)"
    # ganho só nas que sobram da correlação (como no notebook)
    ficha["ganho"] = np.nan
    sobram = ["VP01", "VP02", "VP06", "VP07", "VP08", "VP09", "VP10",
              "faixa_renda", "faixa_idade", "estado"]
    ficha.loc[sobram, "ganho"] = [.2, .15, .1, .01, .3, .005, .1, .05, .04, .045]
    # embaralha a ordem, como a ficha real (ordenada por IV)
    return ficha.sample(frac=1, random_state=1)


@pytest.fixture
def saidas_inventadas():
    return pd.DataFrame({"fica_no_lugar": ["VP01", "VP02"]}, index=["VP04", "VP10"])


def test_funil_reproduz_a_logica_do_notebook(ficha_inventada, saidas_inventadas):
    conjuntos = gerar_conjuntos_do_funil(ficha_inventada, saidas_inventadas)
    todas = conjuntos["todas"][0]
    assert todas[:3] == ["VP01", "VP02", "VP03"] and todas[-3:] == ["faixa_renda", "faixa_idade", "estado"]
    assert len(todas) == 13
    assert "VP03" not in conjuntos["sem_const_inst"][0]
    assert "VP05" not in conjuntos["sem_const_inst"][0]
    assert len(conjuntos["sem_const_inst"][0]) == 11
    sem_corr = conjuntos["sem_corr_090"][0]
    assert "VP04" not in sem_corr and "VP10" not in sem_corr and len(sem_corr) == 9
    # mediana do ganho nas 9 = 0.05; candidatas VP07 (0.01) e VP09 (0.005) saem, VP08 (0.3) fica
    nome_proposta = [n for n in conjuntos if n.startswith("proposta_")][0]
    proposta = conjuntos[nome_proposta][0]
    assert nome_proposta == "proposta_7"
    assert "VP07" not in proposta and "VP09" not in proposta and "VP08" in proposta


def test_funil_para_sem_o_ganho(ficha_inventada, saidas_inventadas):
    ficha_inventada["ganho"] = np.nan
    with pytest.raises(ValueError, match="ganho"):
        gerar_conjuntos_do_funil(ficha_inventada, saidas_inventadas)


def test_funil_para_com_variavel_desconhecida(ficha_inventada):
    saidas = pd.DataFrame({"fica_no_lugar": ["VP01"]}, index=["VP999"])
    with pytest.raises(ValueError, match="fora da ficha"):
        gerar_conjuntos_do_funil(ficha_inventada, saidas)


def test_salvar_conjuntos_do_funil(ficha_inventada, saidas_inventadas, caminho):
    tabela = salvar_conjuntos_do_funil(ficha_inventada, saidas_inventadas, caminho=caminho)
    assert list(tabela["n_variaveis"]) == [13, 11, 9, 7]
    # rodar de novo com as mesmas tabelas não dá erro
    salvar_conjuntos_do_funil(ficha_inventada, saidas_inventadas, caminho=caminho)


# --- conjuntos derivados ------------------------------------------------------------

def test_derivar_troca_na_mesma_posicao(caminho):
    salvar_conjunto("base", ["VP01", "VP07", "VP02"], caminho=caminho)
    novas = derivar_conjunto("m1", "base", trocar={"VP07": "VP05"}, remover=["VP02"],
                             acrescentar=["VP09"], caminho=caminho)
    assert novas == ["VP01", "VP05", "VP09"]
    assert carregar_conjunto("m1", caminho) == novas


def test_derivar_com_variavel_ausente_da_erro(caminho):
    salvar_conjunto("base", ["VP01"], caminho=caminho)
    with pytest.raises(ValueError, match="não estão"):
        derivar_conjunto("x", "base", trocar={"VP07": "VP05"}, caminho=caminho)
