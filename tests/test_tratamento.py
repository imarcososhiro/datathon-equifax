"""Testes do tratamento da base (rodar com `pytest` na raiz do repositório).

[IA] - Testes escritos com auxílio de IA. Usam apenas dados inventados, nunca a base.
"""

import pandas as pd
import pytest

from equifax.tratamento import (ESTADOS, SEM_INFORMACAO, checar_estados, formatar_cep,
                                preencher_sem_informacao,
                                preparar_base)


def base_de_exemplo():
    return pd.DataFrame({
        'cep8': [1310100, 0, 80010000, 69000000],
        'estado': ['São Paulo', 'Sem informação', 'Paraná', 'Amazonas'],
        'faixa_idade': ['26 - 35', None, '36 - 45', '18 - 25'],
        'faixa_renda': ['3 - 1000', '1001 - 2000', '2001 - 3000', '3 - 1000'],
        'adimplencia': [1, 0, 1, 1],
    })


def test_formatar_cep_recupera_o_zero_a_esquerda():
    cep = formatar_cep(pd.Series([1310100, 80010000]))
    assert cep.tolist() == ['01310100', '80010000']


def test_formatar_cep_zero_e_ausente_viram_nulo():
    cep = formatar_cep(pd.Series([0, None, 1310100]))
    assert cep.isna().tolist() == [True, True, False]


def test_estados_sao_os_26_e_o_df():
    assert len(ESTADOS) == len(set(ESTADOS)) == 27
    assert 'Distrito Federal' in ESTADOS


def test_checar_estados_avisa_estado_desconhecido():
    checar_estados(base_de_exemplo())  # 'Sem informação' é aceito
    df = base_de_exemplo()
    df.loc[0, 'estado'] = 'Sao Paulo'  # sem acento: não está na lista
    with pytest.raises(ValueError, match='Sao Paulo'):
        checar_estados(df)


def test_preencher_sem_informacao():
    df = preencher_sem_informacao(base_de_exemplo(), ['faixa_idade'])
    assert df['faixa_idade'].tolist() == ['26 - 35', SEM_INFORMACAO, '36 - 45', '18 - 25']


def test_preparar_base_aplica_os_dois_tratamentos():
    df = preparar_base(base_de_exemplo())
    assert df['cep8'].tolist() == ['01310100', SEM_INFORMACAO, '80010000', '69000000']
    assert df['faixa_idade'].isna().sum() == 0
    # as colunas que não são tratadas continuam iguais
    assert df['adimplencia'].tolist() == [1, 0, 1, 1]


def test_preparar_base_pode_rodar_duas_vezes():
    # Ex.: reaplicar o tratamento numa base já tratada e relida do CSV
    uma_vez = preparar_base(base_de_exemplo())
    duas_vezes = preparar_base(uma_vez.copy())
    pd.testing.assert_frame_equal(uma_vez, duas_vezes)
