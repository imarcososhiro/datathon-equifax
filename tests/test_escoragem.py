"""Testes de src/equifax/escoragem.py com CSVs brutos inventados.

[IA] Testes gerados com auxílio de IA (Claude) em 13/10/2026. Registrar em docs/uso_ia.md.
"""
import json

import numpy as np
import pandas as pd
import pytest

from equifax.escoragem import (ErroEscoragem, carregar_pacote, ensaiar_com_desenvolvimento,
                               escorar_arquivo, salvar_pacote, score_em_pontos)
from equifax.experimentos import escorar, treinar
from equifax.tratamento import preparar_base

VARIAVEIS = ["VP01", "VP02", "VP03", "VP04", "faixa_renda", "faixa_idade", "estado"]


def _bruto(n, safras, semente, com_target=True):
    """CSV bruto no formato das bases: CEP como número (sem o zero à esquerda), nulos."""
    rng = np.random.default_rng(semente)
    df = pd.DataFrame({
        "id": rng.integers(0, 10 * n, n),
        "safra": rng.choice(safras, n),
        "faixa_renda": rng.choice(["3 - 1000", "1001 - 2000", "2001 - 3000"], n),
        "faixa_idade": rng.choice(["18 - 25", "26 - 35", None], n, p=[.45, .45, .1]),
        "estado": rng.choice(["São Paulo", "Bahia", "Sem informação"], n),
        "cep8": rng.choice([1310100, 40020000, 0], n),
        "VP01": rng.integers(0, 50, n).astype(float),
        "VP02": rng.choice([-1, 5, 10], n).astype(float),
        "VP03": rng.normal(size=n).round(3),
        "VP04": rng.integers(0, 5, n).astype(float),
    })
    if com_target:
        logito = 2 + 0.05 * df["VP01"] - 1.0 * df["VP03"]
        df.insert(2, "adimplencia", (rng.random(n) < 1 / (1 + np.exp(-logito))).astype(int))
    return df


@pytest.fixture
def ambiente(tmp_path):
    des = _bruto(3000, range(202301, 202311), 0)
    caminho_des = tmp_path / "base_des.csv"
    des.to_csv(caminho_des, sep=";", index=False)
    tratada = preparar_base(pd.read_csv(caminho_des, sep=";", low_memory=False))
    treino = tratada[tratada["safra"] <= 202308]
    treinado = treinar("lgbm", {"n_estimators": 30}, treino, VARIAVEIS)
    colunas = [c for c in des.columns if c != "adimplencia"]
    caminho_pacote = salvar_pacote(treinado, tmp_path / "pacote.joblib", treino, colunas, "teste")
    oot = _bruto(500, [202311, 202312], 1, com_target=False)
    caminho_oot = tmp_path / "base_oot.csv"
    oot.to_csv(caminho_oot, sep=";", index=False)
    return {"pacote": caminho_pacote, "oot": caminho_oot, "des": caminho_des, "tratada": tratada,
            "treinado": treinado, "tmp": tmp_path}


def test_escora_e_preserva_o_arquivo_original(ambiente):
    saida_csv = ambiente["tmp"] / "saida" / "base_oot_escorada.csv"
    saida, rel = escorar_arquivo(ambiente["oot"], ambiente["pacote"], saida_csv, n_linhas_esperado=500)
    original = pd.read_csv(ambiente["oot"], sep=";", dtype=str, keep_default_na=False)
    lido = pd.read_csv(saida_csv, sep=";", dtype=str, keep_default_na=False)
    assert list(lido.columns) == list(original.columns) + ["score"]
    pd.testing.assert_frame_equal(lido.drop(columns=["score"]), original)  # nada mudou, nem o CEP
    assert lido["score"].astype(float).between(0, 1).all()
    assert rel["linhas"] == 500 and set(rel["safras"]) == {202311, 202312}
    salvo = json.loads(saida_csv.with_suffix(".relatorio.json").read_text(encoding="utf-8"))
    assert salvo["score"]["tipo"] == "probabilidade"


def test_score_igual_ao_calculado_direto(ambiente):
    saida, _ = escorar_arquivo(ambiente["oot"], ambiente["pacote"], gravar=False)
    base = preparar_base(pd.read_csv(ambiente["oot"], sep=";", low_memory=False))
    direto = escorar(carregar_pacote(ambiente["pacote"])["modelo"], base)
    np.testing.assert_allclose(saida["score"].astype(float), np.round(direto, 6))


def test_numero_de_linhas_errado_para(ambiente):
    with pytest.raises(ErroEscoragem, match="linhas"):
        escorar_arquivo(ambiente["oot"], ambiente["pacote"], gravar=False, n_linhas_esperado=12345)


def test_coluna_faltando_para(ambiente):
    oot = pd.read_csv(ambiente["oot"], sep=";").drop(columns=["VP04"])
    caminho = ambiente["tmp"] / "sem_vp04.csv"
    oot.to_csv(caminho, sep=";", index=False)
    with pytest.raises(ErroEscoragem, match="VP04"):
        escorar_arquivo(caminho, ambiente["pacote"], gravar=False)


def test_vp_zerada_numa_safra_vira_alerta(ambiente):
    oot = pd.read_csv(ambiente["oot"], sep=";")
    oot.loc[oot["safra"] == 202312, "VP01"] = 0
    caminho = ambiente["tmp"] / "zerada.csv"
    oot.to_csv(caminho, sep=";", index=False)
    _, rel = escorar_arquivo(caminho, ambiente["pacote"], gravar=False)
    assert rel["vps_zeradas_por_safra"] == [{"variavel": "VP01", "safra": 202312,
                                             "fracao_zeros_safra": 1.0,
                                             "fracao_zeros_treino": pytest.approx(0.02, abs=0.02)}]


def test_categoria_nova_vira_alerta(ambiente):
    oot = pd.read_csv(ambiente["oot"], sep=";")
    oot.loc[:4, "faixa_renda"] = "35001 - 50000"
    caminho = ambiente["tmp"] / "nova.csv"
    oot.to_csv(caminho, sep=";", index=False)
    saida, rel = escorar_arquivo(caminho, ambiente["pacote"], gravar=False)
    assert rel["categorias_novas"] == {"faixa_renda": {"35001 - 50000": 5}}
    assert saida["score"].notna().all()


def test_estado_desconhecido_para(ambiente):
    oot = pd.read_csv(ambiente["oot"], sep=";")
    oot.loc[0, "estado"] = "Atlântida"
    caminho = ambiente["tmp"] / "estado.csv"
    oot.to_csv(caminho, sep=";", index=False)
    with pytest.raises(ValueError, match="Atlântida"):
        escorar_arquivo(caminho, ambiente["pacote"], gravar=False)


def test_transformacao_que_duplica_linhas_para(ambiente, tmp_path):
    pacote = carregar_pacote(ambiente["pacote"])
    pacote["transformacoes"] = [lambda b: pd.concat([b, b.head(1)])]
    with pytest.raises(ErroEscoragem, match="mudou as linhas"):
        escorar_arquivo(ambiente["oot"], pacote, gravar=False)


def test_score_em_pontos(ambiente):
    p = np.array([0.5, 50 / 51, 0.99])
    pontos = score_em_pontos(p)
    assert pontos[1] == 600  # chance de 50 para 1
    assert pontos[0] < pontos[1] < pontos[2]
    saida, rel = escorar_arquivo(ambiente["oot"], ambiente["pacote"], gravar=False,
                                 tipo_score="pontos", nome_coluna="pontuacao")
    assert saida["pontuacao"].dtype.kind == "i" and rel["score"]["nome_coluna"] == "pontuacao"


def test_ensaio_com_desenvolvimento_reproduz_o_score(ambiente):
    resumo = ensaiar_com_desenvolvimento(ambiente["pacote"], ambiente["tratada"], n=400,
                                         caminho_bruto=ambiente["des"], pasta=ambiente["tmp"] / "ensaio")
    assert resumo["linhas"] == 400
    assert resumo["diferenca_maxima"] < 1e-6 and resumo["linhas_diferentes"] == 0
    assert resumo["arquivo_identico_ao_bruto"]


def _tabela_macro():
    """Desocupação inventada: Brasil 8%, São Paulo 7% e Bahia 13% em todos os meses."""
    meses = pd.date_range("2022-01-01", "2023-12-01", freq="MS")
    partes = [pd.DataFrame({"serie": "desocupacao_br", "data_referencia": meses, "uf": "BR",
                            "valor": 8.0, "disponivel_em": meses + pd.DateOffset(months=3)})]
    for uf, valor in [("São Paulo", 7.0), ("Bahia", 13.0)]:
        partes.append(pd.DataFrame({"serie": "desocupacao_uf", "data_referencia": meses, "uf": uf,
                                    "valor": valor, "disponivel_em": meses + pd.DateOffset(months=4)}))
    return pd.concat(partes, ignore_index=True)


def test_pacote_com_features_e_macro(ambiente):
    from equifax.features import Cep, ajustar_familias
    from equifax.macro import JuntarMacro

    tratada = ambiente["tratada"]
    treino, aplicar = ajustar_familias(tratada[tratada["safra"] <= 202308], [Cep(suavizacao=50)])
    juntar = JuntarMacro(_tabela_macro(), ["desocupacao_uf"])
    treino = juntar(treino)
    treinado = treinar("lgbm", {"n_estimators": 30}, treino,
                       VARIAVEIS + ["fe_cep3_taxa_inad", "macro_desocupacao_uf"])
    colunas = [c for c in pd.read_csv(ambiente["des"], sep=";", nrows=0).columns if c != "adimplencia"]
    caminho = salvar_pacote(treinado, ambiente["tmp"] / "pacote_m3.joblib", treino, colunas, "m3",
                            transformacoes=[aplicar, juntar])

    saida, rel = escorar_arquivo(ambiente["oot"], caminho, gravar=False)
    assert saida["score"].notna().all()
    # dado mensal com 4 meses de defasagem: em 01/11 e 01/12 entra o de julho e o de agosto
    assert rel["idade_dados_macro_meses"] == {"desocupacao_uf": {202311: 4, 202312: 4}}

    resumo = ensaiar_com_desenvolvimento(caminho, tratada, n=300, caminho_bruto=ambiente["des"],
                                         pasta=ambiente["tmp"] / "ensaio_m3")
    assert resumo["diferenca_maxima"] < 1e-6 and resumo["arquivo_identico_ao_bruto"]
