"""Motor de experimentos: treina, avalia e registra um modelo num conjunto de variáveis.

[IA] Módulo gerado com auxílio de IA (Claude) em 11/10/2026, a partir das funções
`preparar_x`, `rodar` e `rodar_conjunto` do 02_selecao_variaveis. Registrar em docs/uso_ia.md.

Uso típico:

    base = carregar_desenvolvimento()
    partes = janelas(base)                      # treino e teste (sem OOT interna)
    res = rodar_experimento("lgbm_referencia", "sem_corr_090", partes)
    resumir(ler_registro())

Regras do projeto que o motor garante:
- Nenhum nome de VP no código: a lista vem de `config/conjuntos.yaml` (ou é passada).
- Categorias aprendidas só no treino; categoria nova em outra base vira ausente.
- Avaliar na OOT interna exige `usar_oot_interna=True` (e a parte precisa ter sido
  pedida em `janelas(..., incluir_oot_interna=True)`).
- Cada execução vira uma linha em `reports/experimentos.csv`, só com agregados:
  data, nome, conjunto, modelo, parâmetros, semente, métricas, tempo e commit.

Convenção: adimplencia = 1 é adimplente; o score é a probabilidade de adimplência
(quanto maior, menor o risco), como em `validacao.avaliar`.
"""
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONJUNTOS_PATH, EXPERIMENTOS_PATH, RANDOM_SEED, ROOT
from .conjuntos import carregar_conjunto
from .validacao import avaliar

SEM_INFORMACAO = "Sem informação"
SEMENTES = (RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2)

# Parâmetros de referência (sem ajuste). O LightGBM é o mesmo da seção 2.8 do
# 02_selecao_variaveis; serve de instrumento fixo para comparar conjuntos.
PARAMS_REFERENCIA = {
    "lgbm": dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8),
    "xgb": dict(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=10,
                subsample=0.8, colsample_bytree=0.8),
    "logit_woe": dict(C=1.0, l1_ratio=0.0, iv_minimo=0.02, limiar_correlacao=0.8),
}

# Parâmetros que o motor usa e não passa ao estimador.
# balancear=True: pesos inversos à frequência de cada classe (mais peso aos maus).
PARAMS_DO_MOTOR = ("balancear",)

COLUNAS_REGISTRO = ["data", "nome", "conjunto", "n_variaveis", "modelo", "params", "semente",
                    "avaliado_em", "n_treino", "n_avaliacao", "auc", "gini", "ks",
                    "ks_min_safra", "safra_ks_min", "tempo_treino_s", "commit"]


# --------------------------------------------------------------------------- #
# Modelos disponíveis
# --------------------------------------------------------------------------- #
def _lgbm(params, semente):
    from lightgbm import LGBMClassifier
    base = dict(random_state=semente, n_jobs=-1, verbose=-1, importance_type="gain")
    return LGBMClassifier(**{**base, **params})


def _xgb(params, semente):
    from xgboost import XGBClassifier
    base = dict(random_state=semente, n_jobs=-1, tree_method="hist", enable_categorical=True,
                importance_type="total_gain", eval_metric="auc")
    return XGBClassifier(**{**base, **params})


def _logit_woe(params, semente):
    from .modelos import LogisticaWoE
    return LogisticaWoE(**{"random_state": semente, **params})


# nome -> função (params, semente) -> estimador com fit / predict_proba /
# feature_importances_.
MODELOS = {"lgbm": _lgbm, "xgb": _xgb, "logit_woe": _logit_woe}


def registrar_modelo(nome, construtor, params_referencia=None):
    """Acrescenta um modelo ao motor (ex.: CatBoost, logística com WoE)."""
    MODELOS[nome] = construtor
    if params_referencia is not None:
        PARAMS_REFERENCIA[nome] = dict(params_referencia)


# --------------------------------------------------------------------------- #
# Matriz de entrada
# --------------------------------------------------------------------------- #
def _como_texto(serie):
    """Texto, com o nulo como 'Sem informação' (igual ao 02_selecao_variaveis)."""
    return serie.astype(object).where(serie.notna(), SEM_INFORMACAO).astype(str)


def categorias_do_treino(treino, variaveis):
    """Categorias vistas no treino para cada variável não numérica da lista.

    Devolve dict variável -> lista ordenada de categorias. Só olha o treino.
    """
    return {var: sorted(_como_texto(treino[var]).unique())
            for var in variaveis if not pd.api.types.is_numeric_dtype(treino[var])}


def preparar_x(base, variaveis, categoricas):
    """Colunas `variaveis` de `base`, com as categóricas como 'category'.

    categoricas: dict de `categorias_do_treino` (variável -> categorias do treino).
    Uma categoria que não apareceu no treino vira ausente. As numéricas ficam
    como estão (nulos e códigos especiais -1, -2, 9999 inclusive).
    """
    faltando = [v for v in variaveis if v not in base.columns]
    if faltando:
        raise KeyError(f"Variáveis que não estão na base: {faltando[:10]}")
    x = base[list(variaveis)].copy()
    for var, categorias in categoricas.items():
        if var in x.columns:
            texto = _como_texto(base[var])
            # categoria fora do treino vira ausente antes (o pandas 4 dará erro se não)
            x[var] = pd.Categorical(texto.where(texto.isin(categorias)), categories=categorias)
    return x


# --------------------------------------------------------------------------- #
# Treino, escoragem e avaliação
# --------------------------------------------------------------------------- #
@dataclass
class ModeloTreinado:
    """Um modelo treinado e tudo o que é preciso para escorar outra base igual."""
    estimador: object
    modelo: str
    variaveis: list
    categoricas: dict
    params: dict
    semente: int
    tempo_treino_s: float = field(default=0.0)


def params_do_estimador(params):
    """Tira os parâmetros que são do motor (ex.: balancear) antes de criar o estimador."""
    return {k: v for k, v in params.items() if k not in PARAMS_DO_MOTOR}


def pesos_de_classe(y):
    """Peso de cada linha inverso à frequência da classe (soma igual ao número de linhas)."""
    y = np.asarray(y).astype(int)
    contagem = np.bincount(y, minlength=2)
    return (len(y) / (2 * contagem))[y]


def treinar(modelo, params, treino, variaveis, semente=RANDOM_SEED, col_target="adimplencia"):
    """Treina `modelo` ('lgbm', 'xgb', ...) no treino com as `variaveis` dadas.

    params: dict que completa/substitui os parâmetros de referência do modelo;
        None usa só a referência.
    """
    if modelo not in MODELOS:
        raise KeyError(f"Modelo '{modelo}' não existe. Disponíveis: {list(MODELOS)}")
    params = {**PARAMS_REFERENCIA.get(modelo, {}), **(params or {})}
    variaveis = list(variaveis)
    categoricas = categorias_do_treino(treino, variaveis)
    x = preparar_x(treino, variaveis, categoricas)
    y = treino[col_target].astype(int).to_numpy()

    estimador = MODELOS[modelo](params_do_estimador(params), semente)
    pesos = pesos_de_classe(y) if params.get("balancear") else None
    inicio = time.perf_counter()
    estimador.fit(x, y, sample_weight=pesos)
    tempo = time.perf_counter() - inicio
    if list(estimador.classes_) != [0, 1]:
        raise ValueError(f"Classes inesperadas no target: {list(estimador.classes_)}")
    return ModeloTreinado(estimador, modelo, variaveis, categoricas, params, semente, tempo)


def escorar(treinado, base):
    """Probabilidade de adimplência (score maior = risco menor) para cada linha."""
    x = preparar_x(base, treinado.variaveis, treinado.categoricas)
    return treinado.estimador.predict_proba(x)[:, 1]


def importancia(treinado):
    """Importância de cada variável (ganho), normalizada para somar 1."""
    valores = pd.Series(np.asarray(treinado.estimador.feature_importances_, dtype=float),
                        index=treinado.variaveis)
    total = valores.sum()
    return valores / total if total > 0 else valores


def avaliar_modelo(treinado, base, col_target="adimplencia", col_safra="safra"):
    """Métricas de `avaliar` (total e por safra) na base dada, e a importância."""
    score = escorar(treinado, base)
    return avaliar(base[col_target], score, base[col_safra]), importancia(treinado)


# --------------------------------------------------------------------------- #
# Experimento registrado
# --------------------------------------------------------------------------- #
@dataclass
class ResultadoExperimento:
    registro: pd.DataFrame     # uma linha por semente (o que foi gravado)
    por_safra: pd.DataFrame    # métricas por safra de cada semente
    importancia: pd.Series     # importância média entre as sementes


def commit_atual():
    """Commit curto do Git; '+alteracoes' se há arquivos rastreados modificados."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        sujo = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain",
                               "--untracked-files=no"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("+alteracoes" if sujo else "")
    except (OSError, subprocess.CalledProcessError):
        return "sem_git"


def _gravar_registro(linhas, caminho):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if caminho.exists():
        cabecalho = pd.read_csv(caminho, nrows=0).columns.tolist()
        if cabecalho != COLUNAS_REGISTRO:
            raise ValueError(f"{caminho} tem outras colunas; renomeie o arquivo antigo")
        linhas.to_csv(caminho, mode="a", header=False, index=False, lineterminator="\n")
    else:
        linhas.to_csv(caminho, index=False, lineterminator="\n")


def rodar_experimento(nome, conjunto, partes, modelo="lgbm", params=None, sementes=SEMENTES,
                      avaliar_em="teste", usar_oot_interna=False, registrar=True,
                      caminho_registro=EXPERIMENTOS_PATH, caminho_conjuntos=CONJUNTOS_PATH,
                      nome_conjunto=None, col_target="adimplencia", col_safra="safra"):
    """Treina no treino e avalia em `avaliar_em`, uma vez por semente, e registra.

    nome: rótulo do experimento (ex.: 'lgbm_referencia').
    conjunto: nome de um conjunto de `config/conjuntos.yaml` ou uma lista de
        variáveis (nesse caso, `nome_conjunto` dá o rótulo do registro).
    partes: dict de `dados.janelas` com 'treino' e a parte de avaliação.
    avaliar_em: chave de `partes` usada na avaliação ('teste' por padrão).
        'oot_interna' só com usar_oot_interna=True.
    """
    if avaliar_em == "oot_interna" and not usar_oot_interna:
        raise ValueError("A OOT interna só serve para a confirmação final. "
                         "Para usá-la, passe usar_oot_interna=True.")
    if avaliar_em == "treino":
        raise ValueError("Avaliar no próprio treino não mede generalização")
    if avaliar_em not in partes:
        raise KeyError(f"'{avaliar_em}' não está em partes: {list(partes)}")

    if isinstance(conjunto, str):
        variaveis, rotulo = carregar_conjunto(conjunto, caminho_conjuntos), conjunto
    else:
        variaveis, rotulo = list(conjunto), nome_conjunto or "personalizado"

    treino, avaliacao = partes["treino"], partes[avaliar_em]
    commit = commit_atual()
    linhas, safras, importancias = [], [], []
    for semente in sementes:
        treinado = treinar(modelo, params, treino, variaveis, semente, col_target)
        metricas, imp = avaliar_modelo(treinado, avaliacao, col_target, col_safra)
        total = metricas.iloc[0]
        por_safra = metricas.iloc[1:]
        i_min = por_safra["ks"].idxmin() if por_safra["ks"].notna().any() else None
        linhas.append({
            "data": datetime.now().isoformat(timespec="seconds"),
            "nome": nome, "conjunto": rotulo, "n_variaveis": len(variaveis),
            "modelo": modelo,
            "params": json.dumps(treinado.params, sort_keys=True, ensure_ascii=False),
            "semente": semente, "avaliado_em": avaliar_em,
            "n_treino": len(treino), "n_avaliacao": len(avaliacao),
            "auc": total["auc"], "gini": total["gini"], "ks": total["ks"],
            "ks_min_safra": por_safra.loc[i_min, "ks"] if i_min is not None else np.nan,
            "safra_ks_min": por_safra.loc[i_min, "safra"] if i_min is not None else "",
            "tempo_treino_s": round(treinado.tempo_treino_s, 2), "commit": commit,
        })
        safras.append(metricas.assign(nome=nome, conjunto=rotulo, modelo=modelo, semente=semente))
        importancias.append(imp)

    registro = pd.DataFrame(linhas, columns=COLUNAS_REGISTRO)
    if registrar:
        _gravar_registro(registro, caminho_registro)
    return ResultadoExperimento(registro=registro,
                                por_safra=pd.concat(safras, ignore_index=True),
                                importancia=pd.concat(importancias, axis=1).mean(axis=1))


def ler_registro(caminho=EXPERIMENTOS_PATH):
    """O registro de experimentos inteiro."""
    return pd.read_csv(caminho, dtype={"safra_ks_min": str, "commit": str})


def resumir(registro, nomes=None):
    """Média e desvio entre sementes, por experimento (nome, conjunto, modelo, avaliação).

    Como em `comparar_com_referencia`, só conta a última rodada de cada experimento.
    """
    if nomes is not None:
        registro = registro[registro["nome"].isin(nomes)]
    chaves = ["nome", "conjunto", "n_variaveis", "modelo", "avaliado_em"]
    registro = registro.drop_duplicates(subset=chaves + ["semente"], keep="last")
    return (registro.groupby(chaves, sort=False)[["auc", "gini", "ks", "ks_min_safra",
                                                  "tempo_treino_s"]]
            .agg(["mean", "std"]).round(4))


def treinar_final(modelo, params, base, variaveis, janela=("202301", "202308"), semente=RANDOM_SEED,
                  col_target="adimplencia", col_safra="safra"):
    """Treina o modelo escolhido em todas as linhas das safras da janela (inclusive).

    Decisão em aberto: janela 202301-202308 (deixa a OOT interna para conferir) ou
    202301-202310 (usa as 10 safras, mais perto da OOT cega, sem base para conferir).
    `base` é a base de desenvolvimento inteira; não há divisão em treino e teste aqui.
    """
    inicio, fim = (int(j) for j in janela)
    safra = base[col_safra].astype(int)
    dados = base[(safra >= inicio) & (safra <= fim)]
    if dados.empty:
        raise ValueError(f"Nenhuma linha nas safras {janela}")
    return treinar(modelo, params, dados, variaveis, semente, col_target)


def comparar_com_referencia(registro, referencia, nomes):
    """Ganho de cada experimento sobre a referência, com o desvio entre sementes.

    Uma linha por experimento: AUC, KS e pior KS por safra (médias), as diferenças
    para a referência e o maior desvio entre sementes (da referência ou do próprio).
    `supera_ruido` = o ganho de AUC é maior que esse desvio (critério do spec); fica
    vazio quando não há desvio entre sementes (ex.: logística, que é determinística).

    Só conta a última rodada de cada experimento: quando um notebook roda de novo, o
    registro ganha linhas repetidas com o mesmo nome e semente, e contar as duas
    rodadas encolheria o desvio (foi o que marcou a família de CEP como ganho real
    na segunda rodada do 04, em 13/10).
    """
    r = registro[registro["nome"].isin([referencia, *nomes])]
    r = r.drop_duplicates(subset=["nome", "semente"], keep="last")
    g = r.groupby("nome").agg(n_variaveis=("n_variaveis", "first"), auc=("auc", "mean"),
                              auc_dp=("auc", "std"), ks=("ks", "mean"),
                              ks_min_safra=("ks_min_safra", "mean"))
    base = g.loc[referencia]
    g["delta_auc"] = g["auc"] - base["auc"]
    g["delta_ks"] = g["ks"] - base["ks"]
    g["delta_ks_min_safra"] = g["ks_min_safra"] - base["ks_min_safra"]
    g["ruido"] = np.fmax(g["auc_dp"], base["auc_dp"])
    # Sem variação entre sementes (modelo determinístico, ou uma semente só) não há
    # ruído medido: o critério não se aplica e a coluna fica vazia. Nesse caso, a
    # incerteza que conta é a da amostra de teste, que este critério não mede.
    sem_ruido = g["ruido"].isna() | (g["ruido"] == 0)
    g["supera_ruido"] = (g["delta_auc"] > g["ruido"]).astype(object)
    g.loc[sem_ruido, "supera_ruido"] = None
    return g.loc[[referencia, *nomes]].round(4)


def escolher_vencedor(comparacao, ordem=("M1", "M2", "M3")):
    """Aplica o critério D7 (decidido em 18/10) e devolve (vencedor, tabela com o motivo).

    [IA] Função gerada com auxílio de IA (Claude) em 18/10/2026. Registrar em docs/uso_ia.md.

    `comparacao` tem uma linha por modelo (índice = nome) e as colunas:
    auc_teste, auc_dp_teste, ks_min_safra_teste, ks_min_safra_dp_teste,
    auc_temporal e auc_dp_temporal (médias e desvios entre sementes).

    Critério D7: o desempenho decide quando a diferença supera o ruído; a estabilidade
    é medida pelo pior KS entre as safras; a explicabilidade (aqui, a simplicidade:
    menos variáveis e nenhuma fonte externa) só desempata. Na prática:

    - os modelos são percorridos do mais simples para o mais complexo (`ordem`);
    - um modelo toma o lugar do atual só se ganhar de AUC acima do ruído (maior desvio
      entre sementes dos dois) no teste por cliente E na validação temporal, e se o pior
      KS entre as safras do teste não cair mais que o ruído do pior KS;
    - senão, fica o atual: entre empatados, vence o mais simples.

    A OOT interna não entra aqui: ela só confirma a escolha depois.
    """
    obrigatorias = ["auc_teste", "auc_dp_teste", "ks_min_safra_teste", "ks_min_safra_dp_teste",
                    "auc_temporal", "auc_dp_temporal"]
    faltando = [c for c in obrigatorias if c not in comparacao.columns]
    if faltando:
        raise ValueError(f"Faltam colunas na comparação: {faltando}")
    ordem = [m for m in ordem if m in comparacao.index]
    if not ordem:
        raise ValueError("Nenhum modelo da ordem está na comparação")
    c = comparacao.astype({k: float for k in obrigatorias}).fillna({k: 0.0 for k in obrigatorias})
    atual = ordem[0]
    linhas = [{"modelo": atual, "contra": "—", "ganho_auc_teste": np.nan, "ganho_auc_temporal": np.nan,
               "queda_pior_ks": np.nan, "fica": True, "motivo": "ponto de partida (o mais simples)"}]
    for candidato in ordem[1:]:
        a, b = c.loc[atual], c.loc[candidato]
        ganho_teste = b["auc_teste"] - a["auc_teste"]
        ganho_temporal = b["auc_temporal"] - a["auc_temporal"]
        ruido_teste = max(a["auc_dp_teste"], b["auc_dp_teste"])
        ruido_temporal = max(a["auc_dp_temporal"], b["auc_dp_temporal"])
        queda_ks = a["ks_min_safra_teste"] - b["ks_min_safra_teste"]
        ruido_ks = max(a["ks_min_safra_dp_teste"], b["ks_min_safra_dp_teste"])
        supera_teste = ganho_teste > ruido_teste
        supera_temporal = ganho_temporal > ruido_temporal
        estavel = queda_ks <= ruido_ks
        fica = bool(supera_teste and supera_temporal and estavel)
        if fica:
            motivo = "ganha acima do ruído no teste e na validação temporal, sem perder estabilidade"
        elif not supera_teste or not supera_temporal:
            onde = [n for n, ok in [("no teste", supera_teste), ("na validação temporal", supera_temporal)]
                    if not ok]
            motivo = f"não ganha acima do ruído {' nem '.join(onde)}: empate, fica o mais simples ({atual})"
        else:
            motivo = f"ganha em AUC, mas o pior KS entre safras cai além do ruído: fica {atual}"
        linhas.append({"modelo": candidato, "contra": atual, "ganho_auc_teste": ganho_teste,
                       "ganho_auc_temporal": ganho_temporal, "queda_pior_ks": queda_ks,
                       "fica": fica, "motivo": motivo})
        if fica:
            atual = candidato
    return atual, pd.DataFrame(linhas).set_index("modelo")
