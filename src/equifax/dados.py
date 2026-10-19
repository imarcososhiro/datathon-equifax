"""Leitura da base de desenvolvimento tratada e janelas de validação.

[IA] Módulo gerado com auxílio de IA (Claude) em 11/10/2026. Registrar em docs/uso_ia.md.

A base tratada pelo 01_eda (`data/processed/base_des_01.csv`, cerca de 810 MB)
leva cerca de 1 minuto para ser lida. `carregar_desenvolvimento` lê o CSV uma
vez e guarda uma cópia em parquet em `data/interim/`, com tipos menores:

- VPs em float32, mas só quando a conversão não muda nenhum valor. Uma coluna
  em que algum valor mudaria fica em float64 (a lista vai para o aviso).
- `faixa_renda`, `faixa_idade` e `estado` como categoria.
- `cep8` como texto (os zeros à esquerda continuam lá).

A cópia é refeita sozinha quando o CSV é mais novo que ela.
A OOT interna só sai de `janelas` com `incluir_oot_interna=True`.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DES_TRATADA_PARQUET, DES_TRATADA_PATH
from .validacao import dividir_treino_teste, separar_janelas

COLUNAS_CATEGORIA = ["faixa_renda", "faixa_idade", "estado"]
COLUNAS_TEXTO = ["cep8"]
COLUNAS_CHAVE = ["id", "safra", "adimplencia"]


def _ler_csv(caminho):
    tipos = {c: "string" for c in COLUNAS_TEXTO}
    try:
        return pd.read_csv(caminho, dtype=tipos, engine="pyarrow")
    except (ImportError, ValueError):
        return pd.read_csv(caminho, dtype=tipos, low_memory=False)


def reduzir_tipos(df):
    """Troca float64 por float32 (só se nenhum valor mudar) e texto por categoria.

    Altera o próprio DataFrame. Devolve (df, colunas mantidas em float64).
    """
    mantidas = []
    for col in df.columns:
        if col in COLUNAS_CHAVE:
            continue
        serie = df[col]
        if col in COLUNAS_CATEGORIA:
            df[col] = serie.astype("category")
        elif col in COLUNAS_TEXTO:
            df[col] = serie.astype("string")
        elif pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):
            valores = serie.to_numpy(dtype="float64")
            reduzida = valores.astype("float32")
            if np.array_equal(reduzida.astype("float64"), valores, equal_nan=True):
                df[col] = reduzida
            else:
                df[col] = valores
                mantidas.append(col)
    return df, mantidas


def carregar_desenvolvimento(caminho_csv=DES_TRATADA_PATH, caminho_parquet=DES_TRATADA_PARQUET,
                             recriar=False):
    """Base de desenvolvimento tratada, lida do parquet (ou criada a partir do CSV)."""
    caminho_csv, caminho_parquet = Path(caminho_csv), Path(caminho_parquet)
    atualizado = (caminho_parquet.exists()
                  and (not caminho_csv.exists()
                       or caminho_parquet.stat().st_mtime >= caminho_csv.stat().st_mtime))
    if atualizado and not recriar:
        return pd.read_parquet(caminho_parquet)

    df, mantidas = reduzir_tipos(_ler_csv(caminho_csv))
    if mantidas:
        warnings.warn(f"{len(mantidas)} colunas ficaram em float64 para não mudar valores: "
                      f"{mantidas[:10]}{' ...' if len(mantidas) > 10 else ''}")
    caminho_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(caminho_parquet, index=False)
    return df


def janelas(base, incluir_oot_interna=False):
    """Divide a base nas janelas do Passo 1: {'treino', 'teste'} e, se pedida, 'oot_interna'.

    A OOT interna (202309-202310) só entra com `incluir_oot_interna=True`,
    para a confirmação final; nenhuma decisão de modelo ou de variável usa ela.
    """
    dev, oot_interna = separar_janelas(base)
    treino, teste = dividir_treino_teste(dev)
    partes = {"treino": treino, "teste": teste}
    if incluir_oot_interna:
        partes["oot_interna"] = oot_interna
    return partes
