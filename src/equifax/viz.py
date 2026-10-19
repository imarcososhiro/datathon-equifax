"""Tabelas e gráficos da análise exploratória (Passo 2 do roteiro).

As funções vieram do `notebooks/01_eda.ipynb` sem mudança de comportamento; só a
pasta das figuras passou a vir de `config.FIGURES`. Todo gráfico é salvo em
`reports/figures/` para ser reaproveitado no relatório.

[IA] - As três funções de tabela e gráfico (`distribuicao_frequencia`,
`cruzar_com_adimplencia` e `cruzar_com_renda_e_adimplencia`) foram montadas com
auxílio de IA a partir do código que a equipe tinha escrito célula a célula.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

from equifax.config import FIGURES


def salvar_figura(nome, dpi=150):
    # Salva a figura atual em reports/figures/<nome>.png
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES / f'{nome}.png', dpi=dpi, bbox_inches='tight')


def distribuicao_frequencia(df, variavel, titulo, rotulo, nome_figura, ordem=None, horizontal=False):
    # rotulo: nome da variável no eixo y
    # ordem: lista com a ordem das categorias. Se não for passada, usa a ordem alfabética.
    # horizontal=True: barras deitadas, para variáveis com muitas categorias

    # Tabela
    tabela_frequencia = df[variavel].value_counts()

    if ordem is not None:
        tabela_frequencia = tabela_frequencia.reindex(ordem)
    else:
        tabela_frequencia = tabela_frequencia.sort_index()

    frequencia_relativa = tabela_frequencia / len(df)

    tabela_freq = pd.DataFrame({'Contagem': tabela_frequencia, 'Frequência Relativa': frequencia_relativa})
    display(tabela_freq.style.format({'Frequência Relativa': '{:.2%}'}))

    # Gráfico de barras
    categorias = tabela_frequencia.index.astype(str)
    rotulos_barras = [f'{p:.1%}' for p in frequencia_relativa]

    if horizontal:
        plt.figure(figsize=(9, 10))
        ax = sns.barplot(x=tabela_frequencia.values, y=categorias, color='steelblue')
        ax.bar_label(ax.containers[0], labels=rotulos_barras, padding=3)
        ax.set_xlim(0, tabela_frequencia.max() * 1.12)
        plt.xlabel('Número de clientes')
        plt.ylabel(rotulo)
        plt.grid(axis='x', color="gray", alpha=.5)
    else:
        plt.figure(figsize=(9, 7))
        ax = sns.barplot(x=categorias, y=tabela_frequencia.values, color='steelblue')
        ax.bar_label(ax.containers[0], labels=rotulos_barras)
        ax.set_ylim(0, tabela_frequencia.max() * 1.1)
        plt.xlabel(rotulo)
        plt.ylabel('Número de clientes')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', color="gray", alpha=.5)

    plt.title(titulo)
    plt.tight_layout()
    salvar_figura(nome_figura)
    plt.show()


def cruzar_com_adimplencia(df, variavel, titulo, xlabel, nome_figura, ordem='alfabetica', horizontal=False):

    # ordem: 'alfabetica'  -> categorias em ordem alfabética (padrão)
    #        'decrescente' -> da maior para a menor taxa de inadimplência
    #        lista         -> na ordem da lista (ex.: ORDEM_RENDAS)
    # horizontal=True: barras deitadas, para variáveis com muitas categorias

    # Tabela - Quantidade e % de inadimplentes e adimplentes dentro de cada categoria
    freq_abs = pd.crosstab(df[variavel], df['adimplencia'])
    freq_rel = pd.crosstab(df[variavel], df['adimplencia'], normalize='index') * 100

    # Com 'alfabetica' o crosstab já retorna as categorias em ordem alfabética
    if ordem == 'decrescente':
        ordem = freq_rel[0].sort_values(ascending=False).index.tolist()

    if isinstance(ordem, list):
        freq_abs = freq_abs.reindex(ordem)
        freq_rel = freq_rel.reindex(ordem)

    tabela = freq_abs.astype(str) + ' (' + freq_rel.round(2).astype(str) + '%)'
    tabela['Total'] = freq_abs[0] + freq_abs[1]
    display(tabela.rename(columns={0: 'Inadimplente', 1: 'Adimplente'}))

    # Gráfico de barras da taxa de inadimplência
    taxa = freq_rel[0]
    media = (df['adimplencia'] == 0).mean() * 100

    if horizontal:
        rotulos = [f'{categoria} (n={n})' for categoria, n in zip(taxa.index, tabela['Total'])]

        plt.figure(figsize=(9, 10))
        ax = sns.barplot(x=taxa.values, y=rotulos, color='steelblue')
        ax.bar_label(ax.containers[0], fmt='%.1f%%', padding=3)
        ax.set_xlim(0, taxa.max() * 1.15)

        ax.axvline(media, color='gray', linestyle='--', label=f'Média geral ({media:.1f}%)')

        plt.xlabel('Taxa de inadimplência (%)')
        plt.ylabel(xlabel)
        plt.grid(axis='x', color="gray", alpha=.5)

    else:
        rotulos = [f'{categoria}\n(n={n})' for categoria, n in zip(taxa.index, tabela['Total'])]

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x=rotulos, y=taxa.values, color='steelblue')
        ax.bar_label(ax.containers[0], fmt='%.1f%%')
        ax.set_ylim(0, taxa.max() * 1.15)

        ax.axhline(media, color='gray', linestyle='--', label=f'Média geral ({media:.1f}%)')

        plt.xticks(rotation=45, ha='right')
        plt.xlabel(xlabel)
        plt.ylabel('Taxa de inadimplência (%)')
        plt.grid(axis='y', color="gray", alpha=.5)

    ax.legend()
    plt.title(titulo)
    plt.tight_layout()
    salvar_figura(nome_figura)
    plt.show()


def cruzar_com_renda_e_adimplencia(df, variavel, titulo, rotulo, nome_figura, ordem_rendas, ordem=None,
                                   n_minimo=30):
    # Taxa de inadimplência (%) em cada combinação de variavel (linhas) e faixa de renda (colunas)
    # rotulo: nome da variável no eixo y
    # ordem_rendas: lista com a ordem das faixas de renda (colunas)
    # ordem: lista com a ordem das categorias de variavel (linhas). Se não for passada, usa a ordem alfabética.
    # n_minimo: células com menos clientes que isso ficam em cinza, sem cor de taxa. Com poucos
    #   clientes, um ou dois inadimplentes a mais mudam muito a taxa, então ela não deve ser lida
    #   como risco da combinação.
    # [IA] - Número de clientes por célula e máscara de células pequenas acrescentados com auxílio de IA (11/10).

    inadimplente = df['adimplencia'] == 0

    # Tabelas: taxa e número de clientes de cada combinação
    taxa = pd.crosstab(df[variavel], df['faixa_renda'], values=inadimplente, aggfunc='mean') * 100
    clientes = pd.crosstab(df[variavel], df['faixa_renda'])
    taxa = taxa.reindex(columns=ordem_rendas)
    clientes = clientes.reindex(columns=ordem_rendas).fillna(0).astype(int)

    if ordem is not None:
        taxa = taxa.reindex(index=ordem)
        clientes = clientes.reindex(index=ordem).fillna(0).astype(int)

    display(taxa.style.format('{:.1f}%', na_rep='-'))
    pequenas = (clientes > 0) & (clientes < n_minimo)
    print(f'{int(pequenas.sum().sum())} de {clientes.size} combinações têm menos de {n_minimo} clientes '
          f'(em cinza no gráfico).')

    # Texto de cada célula: taxa e, embaixo, o número de clientes
    textos = taxa.copy().astype(object)
    for linha in taxa.index:
        for coluna in taxa.columns:
            n = clientes.loc[linha, coluna]
            textos.loc[linha, coluna] = '' if n == 0 else f'{taxa.loc[linha, coluna]:.1f}\n(n={n})'

    rotulos = [f'{categoria} (n={clientes.loc[categoria].sum()})' for categoria in taxa.index]

    plt.figure(figsize=(13, 8))
    ax = plt.gca()
    # Fundo cinza: aparece nas células mascaradas (poucos clientes)
    ax.set_facecolor('#d9d9d9')
    sns.heatmap(taxa, mask=(clientes < n_minimo), annot=False, cmap='Reds', yticklabels=rotulos,
                cbar_kws={'label': 'Taxa de inadimplência (%)'}, ax=ax)
    limite_branco = taxa.where(clientes >= n_minimo).max().max() * .6  # texto branco nas células escuras
    for i, linha in enumerate(taxa.index):
        for j, coluna in enumerate(taxa.columns):
            if textos.loc[linha, coluna] == '':
                continue
            pequena = clientes.loc[linha, coluna] < n_minimo
            cor = '#444444' if pequena or taxa.loc[linha, coluna] < limite_branco else 'white'
            ax.text(j + .5, i + .5, textos.loc[linha, coluna], ha='center', va='center', fontsize=8,
                    color=cor, style='italic' if pequena else 'normal')

    plt.xlabel('Faixa de renda')
    plt.ylabel(rotulo)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(f'{titulo}\n(em cinza: menos de {n_minimo} clientes; a taxa não deve ser lida como risco)',
              fontsize=11)
    plt.tight_layout()
    salvar_figura(nome_figura)
    plt.show()
