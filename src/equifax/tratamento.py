"""Tratamento da base (Passo 3 do roteiro).

As transformações daqui nasceram no `notebooks/01_eda.ipynb` (seções 2.2 a 2.4).
Ficam em módulo para que a base de desenvolvimento e a OOT (Passo 10) recebam
exatamente o mesmo tratamento, pela mesma função: `preparar_base`.

As funções alteram o próprio DataFrame recebido (sem cópia), porque a base de
desenvolvimento ocupa cerca de 1 GB em memória.
"""

import pandas as pd

# Categoria usada no lugar do valor ausente (a ausência tem poder discriminativo)
SEM_INFORMACAO = 'Sem informação'

# Colunas em que o NaN vira a categoria "Sem informação"
COLUNAS_SEM_INFORMACAO = ['cep8', 'faixa_idade']

# Faixas de renda da menor para a maior (ordem de tabelas e gráficos)
ORDEM_RENDAS = ['3 - 1000', '1001 - 2000', '2001 - 3000', '3001 - 5000', '5001 - 7000',
                '7001 - 10000', '10001 - 15000', '15001 - 20000', '20001 - 25000',
                '25001 - 35000']

# Os 26 estados e o Distrito Federal, escritos como na base
ESTADOS = ['Acre', 'Alagoas', 'Amapá', 'Amazonas', 'Bahia', 'Ceará', 'Distrito Federal',
           'Espírito Santo', 'Goiás', 'Maranhão', 'Mato Grosso', 'Mato Grosso do Sul',
           'Minas Gerais', 'Pará', 'Paraíba', 'Paraná', 'Pernambuco', 'Piauí', 'Rio de Janeiro',
           'Rio Grande do Norte', 'Rio Grande do Sul', 'Rondônia', 'Roraima', 'Santa Catarina',
           'São Paulo', 'Sergipe', 'Tocantins']


def checar_estados(df):
    # [IA] - A checagem abaixo foi escrita com IA: um estado escrito de outro jeito viraria
    # uma categoria nova sem aviso, então preferimos parar com um erro que diz qual estado é novo.
    desconhecidos = set(df['estado'].dropna().unique()) - set(ESTADOS) - {SEM_INFORMACAO}
    if desconhecidos:
        raise ValueError(f'Estados fora da lista ESTADOS: {sorted(desconhecidos)}')
    return df


def formatar_cep(serie):
    # O cep8 vem como inteiro no CSV e perde os zeros à esquerda (01310100 vira 1310100).
    # Devolve o CEP como texto de 8 dígitos; cep8 == 0 vira nulo <NA>, e não '00000000'.
    cep = (pd.to_numeric(serie, errors='coerce')
             .astype('Int64')
             .astype('string')
             .str.zfill(8))
    return cep.mask(cep == '00000000')


def preencher_sem_informacao(df, colunas=COLUNAS_SEM_INFORMACAO):
    # Troca o NaN das colunas indicadas pela categoria "Sem informação".
    df[colunas] = df[colunas].fillna(SEM_INFORMACAO)
    return df


def preparar_base(df):
    # [IA] - Função montada com IA para conferir os estados e aplicar, na mesma ordem do
    # 01_eda, os dois tratamentos acima. Serve para a base de desenvolvimento e para a OOT.
    df = checar_estados(df)
    df['cep8'] = formatar_cep(df['cep8'])
    df = preencher_sem_informacao(df)
    return df
