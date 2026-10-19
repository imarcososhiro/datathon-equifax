"""Conjuntos de variáveis nomeados, guardados num arquivo versionado.

[IA] Módulo gerado com auxílio de IA (Claude) em 11/10/2026. Registrar em docs/uso_ia.md.

Regra que evita retrabalho: nenhuma função de modelo escreve nomes de VPs no
código. Todas recebem a lista de variáveis, e as listas ficam em
`config/conjuntos.yaml` (só nomes de colunas, nenhum dado). Quando o funil de
seleção fechar, basta salvar o conjunto final e rodar de novo.

`gerar_conjuntos_do_funil` refaz os conjuntos do 02_selecao_variaveis a partir
das tabelas que ele grava em `reports/tables/` (fora do Git), na mesma lógica
do notebook.
"""
import re
from pathlib import Path

import pandas as pd
import yaml

from .config import CONJUNTOS_PATH

CATEGORICAS_FICHA = ["faixa_renda", "faixa_idade", "estado"]


# --------------------------------------------------------------------------- #
# Leitura e escrita do arquivo
# --------------------------------------------------------------------------- #
def _ler(caminho):
    caminho = Path(caminho)
    if not caminho.exists():
        return {}
    with open(caminho, encoding="utf-8") as f:
        conteudo = yaml.safe_load(f) or {}
    return conteudo.get("conjuntos", {}) or {}


def _escrever(conjuntos, caminho):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    cabecalho = ("# Conjuntos de variáveis do projeto (só nomes de colunas).\n"
                 "# Gerado por src/equifax/conjuntos.py; edite pelas funções, não à mão.\n")
    with open(caminho, "w", encoding="utf-8", newline="\n") as f:
        f.write(cabecalho)
        yaml.safe_dump({"conjuntos": conjuntos}, f, allow_unicode=True, sort_keys=False,
                       default_flow_style=None, width=100)


def listar_conjuntos(caminho=CONJUNTOS_PATH):
    """Tabela com nome, número de variáveis e descrição de cada conjunto."""
    conjuntos = _ler(caminho)
    return pd.DataFrame(
        [{"conjunto": nome, "n_variaveis": len(c["variaveis"]), "descricao": c.get("descricao", "")}
         for nome, c in conjuntos.items()],
        columns=["conjunto", "n_variaveis", "descricao"])


def carregar_conjunto(nome, caminho=CONJUNTOS_PATH):
    """Lista de variáveis do conjunto `nome`, na ordem em que foi salva."""
    conjuntos = _ler(caminho)
    if nome not in conjuntos:
        raise KeyError(f"Conjunto '{nome}' não existe. Disponíveis: {list(conjuntos)}")
    return list(conjuntos[nome]["variaveis"])


def salvar_conjunto(nome, variaveis, descricao="", caminho=CONJUNTOS_PATH, sobrescrever=False):
    """Grava (ou, com sobrescrever=True, troca) um conjunto no arquivo versionado."""
    variaveis = [str(v) for v in variaveis]
    if not re.fullmatch(r"[A-Za-z0-9_]+", str(nome)):
        raise ValueError("O nome do conjunto só pode ter letras, números e _")
    if not variaveis:
        raise ValueError("O conjunto está vazio")
    repetidas = sorted({v for v in variaveis if variaveis.count(v) > 1})
    if repetidas:
        raise ValueError(f"Variáveis repetidas no conjunto: {repetidas}")

    conjuntos = _ler(caminho)
    if nome in conjuntos and not sobrescrever:
        if list(conjuntos[nome]["variaveis"]) == variaveis:
            return  # já está salvo igual
        raise ValueError(f"Conjunto '{nome}' já existe com outra lista; use sobrescrever=True")
    conjuntos[nome] = {"descricao": descricao, "n_variaveis": len(variaveis),
                       "variaveis": variaveis}
    _escrever(conjuntos, caminho)


# --------------------------------------------------------------------------- #
# Conjuntos do funil do 02_selecao_variaveis
# --------------------------------------------------------------------------- #
def _ordem_natural(var):
    """Mesma ordem do 02_selecao_variaveis: VPs pelo número, depois as categóricas.

    A ordem das colunas pode mudar levemente o resultado do LightGBM (empates e
    amostragem de colunas), por isso é a mesma do notebook.
    """
    m = re.fullmatch(r"VP(\d+)", var)
    if m:
        return (0, int(m.group(1)), "")
    if var in CATEGORICAS_FICHA:
        return (1, CATEGORICAS_FICHA.index(var), "")
    return (2, 0, var)


def gerar_conjuntos_do_funil(ficha, saidas_correlacao):
    """Refaz os 4 conjuntos do funil a partir das tabelas do 02_selecao_variaveis.

    ficha: `reports/tables/ficha_variaveis.csv` (índice = variável), com as colunas
        `sugestao`, `classe_psi` e `ganho` (ganho no modelo sem correlacionadas).
    saidas_correlacao: `reports/tables/saidas_correlacao.csv` (índice = variável
        que sai por correlação).

    Mesma lógica das seções 2.7 e 2.8 do notebook:
        todas           -> todas as variáveis da ficha
        sem_const_inst  -> sem constantes (P1) e sem instáveis no PSI (P2)
        sem_corr_090    -> sem uma de cada grupo de correlação >= 0,90 (P7)
        proposta_242    -> sem as de IV < 0,02 com ganho abaixo da mediana (P3)
    Devolve dict nome -> (lista, descrição).
    """
    todas = sorted(ficha.index.astype(str), key=_ordem_natural)
    constantes = set(ficha.index[ficha["sugestao"] == "remover (constante)"])
    instaveis = set(ficha.index[ficha["classe_psi"] == "instável"])
    sem_const_inst = [v for v in todas if v not in constantes | instaveis]

    saem_corr = set(saidas_correlacao.index.astype(str))
    desconhecidas = saem_corr - set(todas)
    if desconhecidas:
        raise ValueError(f"saidas_correlacao tem variáveis fora da ficha: {sorted(desconhecidas)}")
    sem_corr = [v for v in sem_const_inst if v not in saem_corr]

    ganho = ficha.loc[sem_corr, "ganho"]
    if ganho.isna().any():
        raise ValueError("A ficha não tem o ganho de todas as variáveis do conjunto sem "
                         "correlacionadas; rode a seção 2.8 do 02_selecao_variaveis")
    candidatas = [v for v in sem_corr if ficha.loc[v, "sugestao"] == "candidata a remoção (testar)"]
    mediana = ganho.median()
    pouco_usadas = {v for v in candidatas if ganho[v] < mediana}
    proposta = [v for v in sem_corr if v not in pouco_usadas]

    return {
        "todas": (todas, "Todas as variáveis da ficha (VPs e categóricas). Referência."),
        "sem_const_inst": (sem_const_inst, "Sem as constantes (P1) e sem as instáveis no PSI (P2)."),
        "sem_corr_090": (sem_corr, "Sem correlacionadas, limiar 0,90, fica a de maior IV (P7)."),
        f"proposta_{len(proposta)}": (proposta, "Sem as de IV < 0,02 com ganho abaixo da mediana "
                                                "no modelo sem correlacionadas (P3). Conjunto final do funil."),
    }


def salvar_conjuntos_do_funil(ficha, saidas_correlacao, caminho=CONJUNTOS_PATH,
                              sobrescrever=False):
    """Gera os conjuntos do funil e grava no arquivo versionado. Devolve a listagem."""
    for nome, (variaveis, descricao) in gerar_conjuntos_do_funil(ficha, saidas_correlacao).items():
        salvar_conjunto(nome, variaveis, descricao, caminho=caminho, sobrescrever=sobrescrever)
    return listar_conjuntos(caminho)


def derivar_conjunto(nome, origem, trocar=None, remover=(), acrescentar=(), descricao="",
                     caminho=CONJUNTOS_PATH, sobrescrever=False):
    """Cria um conjunto a partir de outro: troca, remove ou acrescenta variáveis.

    trocar: dict {variável antiga: variável nova}; a nova fica na mesma posição.
    Exemplo: trocar={"VP07": "VP05"}.
    Devolve a lista gravada.
    """
    variaveis = carregar_conjunto(origem, caminho)
    trocar = dict(trocar or {})
    faltando = [v for v in list(trocar) + list(remover) if v not in variaveis]
    if faltando:
        raise ValueError(f"Variáveis que não estão em '{origem}': {faltando}")
    novas = [trocar.get(v, v) for v in variaveis if v not in set(remover)]
    novas += [v for v in acrescentar if v not in novas]
    salvar_conjunto(nome, novas, descricao or f"Derivado de {origem}", caminho=caminho,
                    sobrescrever=sobrescrever)
    return novas
