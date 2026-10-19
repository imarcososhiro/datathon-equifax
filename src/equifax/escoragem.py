"""Escoragem da OOT (Passo 10): o modelo escolhido, salvo num pacote, escora um CSV bruto.

[IA] Módulo gerado com auxílio de IA (Claude) em 13/10/2026. Registrar em docs/uso_ia.md.

A OOT cega continua fechada até o Passo 10: este módulo é testado com dados
inventados e com uma amostra da base de desenvolvimento (`ensaiar_com_desenvolvimento`).

Fluxo de `escorar_arquivo`:
1. Lê o CSV bruto duas vezes: como texto (para devolver o arquivo exatamente como
   veio) e com tipos (para o modelo).
2. Aplica o mesmo tratamento da base de desenvolvimento (`tratamento.preparar_base`)
   e, se houver, as transformações extras do modelo (por exemplo, a junção macro do M3).
3. Faz as checagens antes de gravar: colunas, número de linhas, ordem e ids, linhas
   repetidas, categorias novas, VPs zeradas numa safra (falha de carga num mês),
   score sem nulos e entre 0 e 1.
4. Grava o arquivo original com a coluna de score no fim, e um relatório das checagens.

Nome da coluna, separador e tipo de score (probabilidade ou pontos) são parâmetros,
porque a organização ainda não respondeu o formato.
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import DES_PATH, MODELS, SUBMISSION
from .experimentos import commit_atual, escorar
from .tratamento import preparar_base

SEPARADOR_BRUTO = ";"
COL_TARGET = "adimplencia"
CHAVE = ["id", "safra"]


class ErroEscoragem(Exception):
    """Uma checagem obrigatória falhou; nada foi gravado."""


# --------------------------------------------------------------------------- #
# Pacote do modelo
# --------------------------------------------------------------------------- #
def _fracao_zeros(base, variaveis):
    vps = [v for v in variaveis if v in base.columns and pd.api.types.is_numeric_dtype(base[v])]
    return {v: float((base[v] == 0).mean()) for v in vps}


def colunas_brutas_do_desenvolvimento(caminho=DES_PATH, sep=SEPARADOR_BRUTO):
    """Colunas do CSV bruto de desenvolvimento, sem o target (só o cabeçalho é lido)."""
    return [c for c in pd.read_csv(caminho, sep=sep, nrows=0).columns if c != COL_TARGET]


def salvar_pacote(treinado, caminho, treino, colunas_brutas, nome, transformacoes=None,
                  observacoes=""):
    """Grava num arquivo tudo que a escoragem precisa.

    treinado: ModeloTreinado (experimentos.treinar ou treinar_final).
    treino: as linhas usadas no treino, para guardar a fração de zeros de cada VP
        (referência da checagem de VPs zeradas).
    colunas_brutas: colunas esperadas no CSV a escorar (sem o target).
    transformacoes: lista de funções base -> base aplicadas depois do tratamento
        (por exemplo, a junção macro); precisam poder ser gravadas pelo joblib.
    """
    pacote = {
        "modelo": treinado,
        "colunas_brutas": list(colunas_brutas),
        "fracao_zeros_treino": _fracao_zeros(treino, treinado.variaveis),
        "transformacoes": list(transformacoes or []),
        "metadados": {"nome": nome, "modelo": treinado.modelo, "params": treinado.params,
                      "n_variaveis": len(treinado.variaveis), "n_treino": len(treino),
                      "safras_treino": sorted(int(s) for s in pd.unique(treino["safra"])),
                      "criado_em": datetime.now().isoformat(timespec="seconds"),
                      "commit": commit_atual(), "observacoes": observacoes},
    }
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pacote, caminho)
    return caminho


def carregar_pacote(caminho):
    return joblib.load(caminho)


# --------------------------------------------------------------------------- #
# Leitura e checagens
# --------------------------------------------------------------------------- #
def ler_bruto(caminho, sep=SEPARADOR_BRUTO):
    """(texto, base): o CSV como texto, intocado, e o CSV com tipos e tratado."""
    texto = pd.read_csv(caminho, sep=sep, dtype=str, keep_default_na=False)
    base = pd.read_csv(caminho, sep=sep, low_memory=False)
    base = preparar_base(base)
    return texto, base


def checar_vps_zeradas(base, fracao_zeros_treino, limite_treino=0.99):
    """VPs com 100% de zeros numa safra, quando no treino tinham menos de `limite_treino`.

    Pega falhas de carga em que uma VP vem toda zerada num mês.
    """
    alertas = []
    for vp, frac in fracao_zeros_treino.items():
        if vp not in base.columns or frac >= limite_treino:
            continue
        por_safra = (base[vp] == 0).groupby(base["safra"]).mean()
        for safra, f in por_safra.items():
            if f == 1.0:
                alertas.append({"variavel": vp, "safra": int(safra), "fracao_zeros_safra": 1.0,
                                "fracao_zeros_treino": round(frac, 4)})
    return pd.DataFrame(alertas, columns=["variavel", "safra", "fracao_zeros_safra",
                                          "fracao_zeros_treino"])


def checar_categorias_novas(base, treinado):
    """Valores de variáveis categóricas que o modelo não viu no treino (viram ausentes)."""
    novas = {}
    for var, categorias in treinado.categoricas.items():
        valores = base[var].astype(object).where(base[var].notna(), "Sem informação").astype(str)
        fora = valores[~valores.isin(categorias)]
        if len(fora):
            novas[var] = {k: int(v) for k, v in fora.value_counts().items()}
    return novas


def score_em_pontos(probabilidade, pdo=20, pontos_base=600, odds_base=50):
    """Escala de mercado: `pontos_base` = chance de `odds_base` bons para 1 mau; dobra a cada `pdo`."""
    p = np.clip(np.asarray(probabilidade, dtype=float), 1e-6, 1 - 1e-6)
    return np.round(pontos_base + pdo / np.log(2) * (np.log(p / (1 - p)) - np.log(odds_base)))


# --------------------------------------------------------------------------- #
# Escoragem
# --------------------------------------------------------------------------- #
def escorar_arquivo(caminho_entrada, pacote, caminho_saida=None, nome_coluna="score",
                    separador_saida=SEPARADOR_BRUTO, tipo_score="probabilidade",
                    n_linhas_esperado=None, casas_decimais=6, gravar=True):
    """Escora um CSV bruto com o pacote do modelo e grava o arquivo com o score.

    Devolve (tabela com o score, relatório das checagens). Uma checagem obrigatória
    que falha levanta ErroEscoragem e nada é gravado. Alertas (VPs zeradas,
    categorias novas, chaves repetidas) vão para o relatório sem interromper.
    """
    if tipo_score not in ("probabilidade", "pontos"):
        raise ValueError("tipo_score deve ser 'probabilidade' ou 'pontos'")
    if isinstance(pacote, (str, Path)):
        pacote = carregar_pacote(pacote)
    treinado = pacote["modelo"]
    texto, base = ler_bruto(caminho_entrada)
    relatorio = {"entrada": str(caminho_entrada), "pacote": pacote["metadados"],
                 "escorado_em": datetime.now().isoformat(timespec="seconds")}

    # 1. Colunas
    esperadas = pacote["colunas_brutas"]
    faltando = [c for c in esperadas if c not in texto.columns]
    sobrando = [c for c in texto.columns if c not in esperadas]
    if faltando or sobrando:
        raise ErroEscoragem(f"Colunas diferentes do desenvolvimento. Faltando: {faltando[:10]}; "
                            f"sobrando: {sobrando[:10]}")
    # 2. Linhas
    if n_linhas_esperado is not None and len(texto) != n_linhas_esperado:
        raise ErroEscoragem(f"{len(texto)} linhas; esperado {n_linhas_esperado}")
    relatorio["linhas"] = len(texto)
    # 3. Chaves repetidas (só alerta: o id se repete entre e dentro das safras)
    repetidas = int(base.duplicated(subset=CHAVE).sum())
    relatorio["chaves_id_safra_repetidas"] = repetidas
    relatorio["safras"] = {int(k): int(v) for k, v in base["safra"].value_counts().sort_index().items()}

    # 4. Transformações extras (ex.: macro) e checagem de que não mudaram as linhas
    indice = base.index.copy()
    for f in pacote["transformacoes"]:
        base = f(base)
    if not base.index.equals(indice):
        raise ErroEscoragem("Uma transformação mudou as linhas da base (junção duplicou ou perdeu)")
    # Idade dos dados macro usados (meses entre a referência e a concessão), por safra:
    # em produção, uma idade acima da defasagem esperada indica divulgação atrasada.
    idades = [c for c in base.columns if c.endswith("_idade_meses")]
    if idades:
        relatorio["idade_dados_macro_meses"] = {
            c.removeprefix("macro_").removesuffix("_idade_meses"):
                {int(s): (int(v) if pd.notna(v) else None)
                 for s, v in base.groupby("safra")[c].max().items()}
            for c in idades}

    # 5. Alertas de qualidade
    zeradas = checar_vps_zeradas(base, pacote["fracao_zeros_treino"])
    relatorio["vps_zeradas_por_safra"] = zeradas.to_dict(orient="records")
    relatorio["categorias_novas"] = checar_categorias_novas(base, treinado)

    # 6. Score
    prob = escorar(treinado, base)
    if len(prob) != len(texto) or np.isnan(prob).any():
        raise ErroEscoragem("Score com nulos ou com tamanho diferente da entrada")
    if (prob < 0).any() or (prob > 1).any():
        raise ErroEscoragem("Probabilidade fora de [0, 1]")
    score = prob if tipo_score == "probabilidade" else score_em_pontos(prob)
    relatorio["score"] = {"tipo": tipo_score, "nome_coluna": nome_coluna,
                          "media": float(np.mean(score)), "min": float(np.min(score)),
                          "max": float(np.max(score))}

    # 7. Saída: o arquivo original, intocado, com a coluna de score no fim
    if nome_coluna in texto.columns:
        raise ErroEscoragem(f"A entrada já tem uma coluna chamada {nome_coluna}")
    saida = texto.copy()
    saida[nome_coluna] = (np.round(score, casas_decimais) if tipo_score == "probabilidade"
                          else score.astype(int))
    if not (saida["id"].to_numpy() == base["id"].astype(str).to_numpy()).all():
        raise ErroEscoragem("A ordem das linhas mudou entre a leitura e a saída")

    if gravar:
        caminho_saida = Path(caminho_saida or SUBMISSION / "base_oot_escorada.csv")
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        saida.to_csv(caminho_saida, sep=separador_saida, index=False, lineterminator="\n")
        relatorio["saida"] = str(caminho_saida)
        with open(caminho_saida.with_suffix(".relatorio.json"), "w", encoding="utf-8") as f:
            json.dump(relatorio, f, ensure_ascii=False, indent=2, default=str)
    return saida, relatorio


# --------------------------------------------------------------------------- #
# Ensaio com a base de desenvolvimento (a OOT continua fechada)
# --------------------------------------------------------------------------- #
def ensaiar_com_desenvolvimento(pacote, base_tratada, n=5000, safras=None, semente=0,
                                caminho_bruto=DES_PATH, pasta=None):
    """Escora pelo caminho do CSV bruto uma amostra do desenvolvimento e compara.

    Sorteia `n` linhas do CSV bruto de desenvolvimento, tira o target, grava como se
    fosse uma OOT e escora com `escorar_arquivo`. Compara com o score das mesmas
    linhas calculado direto na base tratada (`base_tratada`, a do 01_eda).
    Diferença zero mostra que o caminho bruto -> tratamento -> score reproduz o
    desenvolvimento. Devolve um resumo só com agregados.
    """
    pasta = Path(pasta or MODELS / "ensaio_escoragem")
    pasta.mkdir(parents=True, exist_ok=True)
    bruto = pd.read_csv(caminho_bruto, sep=SEPARADOR_BRUTO, dtype=str, keep_default_na=False)
    if len(bruto) != len(base_tratada):
        raise ErroEscoragem("A base tratada não tem as mesmas linhas do CSV bruto")
    candidatas = np.arange(len(bruto))
    if safras is not None:
        candidatas = candidatas[bruto["safra"].astype(int).isin(list(safras)).to_numpy()]
    rng = np.random.default_rng(semente)
    posicoes = np.sort(rng.choice(candidatas, size=min(n, len(candidatas)), replace=False))
    amostra = bruto.iloc[posicoes].reset_index(drop=True)
    # a base tratada (01_eda) mantém a ordem do CSV bruto: confere linha a linha
    linhas = base_tratada.iloc[posicoes].reset_index(drop=True)
    if not ((linhas["id"].astype(int).to_numpy() == amostra["id"].astype(int).to_numpy()).all()
            and (linhas["safra"].astype(int).to_numpy() == amostra["safra"].astype(int).to_numpy()).all()):
        raise ErroEscoragem("A base tratada não está na mesma ordem do CSV bruto")

    entrada = pasta / "amostra_sem_target.csv"
    amostra.drop(columns=[COL_TARGET]).to_csv(entrada, sep=SEPARADOR_BRUTO, index=False,
                                             lineterminator="\n")
    saida, relatorio = escorar_arquivo(entrada, pacote, pasta / "amostra_escorada.csv",
                                       n_linhas_esperado=len(amostra))
    if isinstance(pacote, (str, Path)):
        pacote = carregar_pacote(pacote)
    for f in pacote["transformacoes"]:  # features e macro, como na escoragem
        linhas = f(linhas)
    direto = escorar(pacote["modelo"], linhas)
    diferenca = np.abs(saida["score"].astype(float).to_numpy() - np.round(direto, 6))
    return {"linhas": len(amostra), "diferenca_maxima": float(diferenca.max()),
            "linhas_diferentes": int((diferenca > 1e-6).sum()),
            "vps_zeradas": len(relatorio["vps_zeradas_por_safra"]),
            "categorias_novas": relatorio["categorias_novas"],
            "arquivo_identico_ao_bruto": bool(
                saida.drop(columns=["score"]).equals(amostra.drop(columns=[COL_TARGET])))}
