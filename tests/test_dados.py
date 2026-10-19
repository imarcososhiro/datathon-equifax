"""Testes de src/equifax/dados.py com uma base inventada gravada em pasta temporária.

[IA] Testes gerados com auxílio de IA (Claude) em 11/10/2026. Registrar em docs/uso_ia.md.
"""
import os

import numpy as np
import pandas as pd
import pytest

from equifax.dados import carregar_desenvolvimento, janelas, reduzir_tipos


@pytest.fixture
def caminhos(tmp_path, base):
    csv = tmp_path / "processed" / "base.csv"
    csv.parent.mkdir()
    base.to_csv(csv, index=False)
    return csv, tmp_path / "interim" / "base.parquet"


def test_reduzir_tipos_so_troca_quando_nada_muda():
    df = pd.DataFrame({"id": [1, 2], "safra": [202301, 202302], "adimplencia": [1, 0],
                       "VP01": [1.0, 9999.0],                # cabe em float32
                       "VP02": [0.1, 123456789.123],         # mudaria em float32
                       "faixa_renda": ["3 - 1000", None], "cep8": ["01310100", "Sem informação"]})
    df, mantidas = reduzir_tipos(df)
    assert df["VP01"].dtype == np.float32
    assert df["VP02"].dtype == np.float64 and mantidas == ["VP02"]
    assert df["faixa_renda"].dtype == "category"
    assert df["cep8"].iloc[0] == "01310100"
    assert df["id"].dtype.kind == "i"  # chave fica como estava


@pytest.mark.filterwarnings("ignore:.*float64")
def test_carregar_cria_parquet_igual_ao_csv(caminhos, base):
    csv, parquet = caminhos
    df = carregar_desenvolvimento(csv, parquet)
    assert parquet.exists()
    assert df.shape == base.shape
    assert (df["cep8"].astype(str) == base["cep8"]).all()  # zeros à esquerda preservados
    for var in ["VP03", "VP06"]:  # nenhum valor muda, com ou sem decimais
        np.testing.assert_array_equal(df[var].astype("float64"), base[var])
    assert df["VP06"].dtype == np.float32  # inteiros
    assert df["VP03"].dtype == np.float64  # decimais que mudariam em float32
    # Segunda leitura vem do parquet e é igual à primeira
    de_novo = carregar_desenvolvimento(csv, parquet)
    pd.testing.assert_frame_equal(df, de_novo)


@pytest.mark.filterwarnings("ignore:.*float64")
def test_parquet_e_refeito_quando_o_csv_muda(caminhos, base):
    csv, parquet = caminhos
    carregar_desenvolvimento(csv, parquet)
    base.iloc[:100].to_csv(csv, index=False)
    futuro = parquet.stat().st_mtime + 10
    os.utime(csv, (futuro, futuro))
    assert len(carregar_desenvolvimento(csv, parquet)) == 100


@pytest.mark.filterwarnings("ignore:.*float64")
def test_recriar_forca_nova_leitura(caminhos, base):
    csv, parquet = caminhos
    carregar_desenvolvimento(csv, parquet)
    base.iloc[:50].to_csv(csv, index=False)
    os.utime(csv, (0, 0))  # CSV "mais velho" que o parquet
    assert len(carregar_desenvolvimento(csv, parquet)) == len(base)
    assert len(carregar_desenvolvimento(csv, parquet, recriar=True)) == 50


def test_janelas_sem_oot_interna_por_padrao(base):
    partes = janelas(base)
    assert set(partes) == {"treino", "teste"}
    assert not set(partes["treino"]["id"]) & set(partes["teste"]["id"])
    assert partes["treino"]["safra"].max() <= 202308


def test_janelas_com_oot_interna_so_quando_pedida(base):
    partes = janelas(base, incluir_oot_interna=True)
    assert set(partes["oot_interna"]["safra"]) == {202309, 202310}
