"""Métricas de seleção de variáveis: WoE, IV e PSI.

[IA] Módulo gerado com auxílio de IA (Claude). Registrar em docs/uso_ia.md.

Regras (Passos 2 e 4):
- Tudo é calculado SÓ no treino (202301-202308, fold de treino). Teste, OOT
  interna e OOT cega não entram.
- Códigos especiais (-1, -2, 9999) e NaN ficam em faixas próprias: não são
  misturados com os valores "normais" nem convertidos em NaN.
- Convenção de WoE (Siddiqi): WoE = ln(%bons / %maus). WoE > 0 => faixa com
  mais bons. Para isso o optbinning recebe o MAU como evento (y = 1 - adimplencia).
- Este módulo só MEDE e CLASSIFICA. Nenhuma variável é removida aqui; o corte
  final depende de ablação (Passo 4).
"""
import warnings

import numpy as np
import pandas as pd
from optbinning import OptimalBinning

CODIGOS_ESPECIAIS = [-1, -2, 9999]
EPS = 1e-4  # suavização para evitar log(0)
SEM_INFORMACAO = "Sem informação"


# --------------------------------------------------------------------------- #
# WoE e IV
# --------------------------------------------------------------------------- #
def ajustar_binning(x, adimplencia, tipo="numerical",
                    codigos_especiais=CODIGOS_ESPECIAIS, **kwargs):
    """Ajusta um binning ótimo (optbinning) para uma variável.

    x: Series da variável (no treino)
    adimplencia: Series 1 = bom, 0 = mau
    tipo: 'numerical' para VPs, 'categorical' para faixas e estado
    Retorna o objeto OptimalBinning ajustado.
    """
    y_mau = (1 - adimplencia).to_numpy()
    # Cada código especial presente vira uma faixa separada (-1 e -2 podem
    # significar coisas diferentes, então não são juntados num "Special" só)
    especiais = None
    if tipo == "numerical":
        presentes = [c for c in codigos_especiais if (x == c).any()]
        especiais = {f"esp_{c}": [c] for c in presentes} or None
    params = dict(name=str(x.name), dtype=tipo, special_codes=especiais,
                  min_prebin_size=0.02, max_n_prebins=20, monotonic_trend="auto")
    params.update(kwargs)
    ob = OptimalBinning(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ob.fit(x.to_numpy(), y_mau)
    return ob


def tabela_woe(ob):
    """Tabela por faixa: contagens, taxa de inadimplência, WoE e IV."""
    t = ob.binning_table.build(show_digits=4).copy()
    t = t[~t["Bin"].astype(str).isin(["", "Totals"])]
    t = t.rename(columns={"Count": "total", "Non-event": "bons", "Event": "maus",
                          "Event rate": "taxa_inadimplencia"})
    t = t[t["total"] > 0]  # tira as faixas "Special" e "Missing" vazias
    return t[["Bin", "total", "bons", "maus", "taxa_inadimplencia", "WoE", "IV"]]


def iv_de(ob):
    t = ob.binning_table.build()
    return float(t.loc["Totals", "IV"])


def woe_por_categoria(x, adimplencia):
    """WoE e IV de cada categoria, sem agrupar (uma faixa por categoria).

    Complementa o binning ótimo, que junta categorias parecidas: aqui dá para
    ver cada faixa separada, por exemplo para decidir fusões de faixas vizinhas.
    """
    dados = pd.DataFrame({"categoria": _como_categoria(x).to_numpy(),
                          "bom": np.asarray(adimplencia)})
    t = dados.groupby("categoria")["bom"].agg(total="count", bons="sum")
    t["maus"] = t["total"] - t["bons"]
    t["taxa_inadimplencia"] = t["maus"] / t["total"]
    pct_bons = (t["bons"] / t["bons"].sum()).clip(lower=EPS)
    pct_maus = (t["maus"] / t["maus"].sum()).clip(lower=EPS)
    t["WoE"] = np.log(pct_bons / pct_maus)
    t["IV"] = (pct_bons - pct_maus) * t["WoE"]
    return t


def classificar_iv(iv):
    """Faixas usuais de mercado (Siddiqi)."""
    if iv < 0.02:
        return "sem poder"
    if iv < 0.10:
        return "fraco"
    if iv < 0.30:
        return "médio"
    if iv < 0.50:
        return "forte"
    return "suspeito (revisar)"


# --------------------------------------------------------------------------- #
# PSI
# --------------------------------------------------------------------------- #
def _como_categoria(x):
    """Texto para variáveis categóricas, com o nulo como 'Sem informação'."""
    return x.astype(object).where(x.notna(), SEM_INFORMACAO).astype(str)


def _faixas_psi(ref, n_bins=10, codigos_especiais=CODIGOS_ESPECIAIS):
    """Cortes por quantil da referência, ignorando especiais e NaN.

    As pontas são -inf e +inf: valores fora do intervalo da referência
    continuam sendo contados (np.histogram com cortes fixos os descartaria).
    Devolve None quando cada valor deve virar uma faixa: variável de texto
    ou numérica com poucos valores distintos.
    """
    if not pd.api.types.is_numeric_dtype(ref):
        return None
    normais = ref[~ref.isin(codigos_especiais) & ref.notna()]
    if normais.nunique() <= n_bins:
        return None
    cortes = np.unique(np.quantile(normais, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.concatenate([[-np.inf], cortes, [np.inf]])


def _distribuicao(x, cortes, codigos_especiais=CODIGOS_ESPECIAIS):
    """Proporção de x em cada faixa, com especiais e NaN em faixas próprias."""
    rotulos = pd.Series(index=x.index, dtype=object)
    if pd.api.types.is_numeric_dtype(x):
        for c in codigos_especiais:
            rotulos[x == c] = f"esp_{c}"
    rotulos[x.isna()] = "NaN"
    normais = rotulos.isna()
    if cortes is None:
        rotulos[normais] = x[normais].astype(str)
    else:
        rotulos[normais] = pd.cut(x[normais], cortes).astype(str)
    return rotulos.value_counts(normalize=True)


def _psi_de_distribuicoes(p_ref, p_atu):
    faixas = p_ref.index.union(p_atu.index)
    p_ref = p_ref.reindex(faixas, fill_value=0).clip(lower=EPS)
    p_atu = p_atu.reindex(faixas, fill_value=0).clip(lower=EPS)
    return float(((p_atu - p_ref) * np.log(p_atu / p_ref)).sum())


def psi(ref, atual, n_bins=10, codigos_especiais=CODIGOS_ESPECIAIS):
    """PSI = soma (%atual - %ref) * ln(%atual / %ref)."""
    cortes = _faixas_psi(ref, n_bins, codigos_especiais)
    return _psi_de_distribuicoes(_distribuicao(ref, cortes, codigos_especiais),
                                 _distribuicao(atual, cortes, codigos_especiais))


def _psi_por_safra(x, safra, safras_ref=None, n_bins=10,
                   codigos_especiais=CODIGOS_ESPECIAIS):
    """PSI de cada safra contra a referência; x e safra são Series alinhadas."""
    safras = sorted(safra.unique())
    safras_ref = list(safras_ref) if safras_ref is not None else safras[:2]
    ref = x[safra.isin(safras_ref)]
    cortes = _faixas_psi(ref, n_bins, codigos_especiais)
    p_ref = _distribuicao(ref, cortes, codigos_especiais)  # calculada uma vez só
    return pd.Series(
        {s: _psi_de_distribuicoes(p_ref, _distribuicao(x[safra == s], cortes, codigos_especiais))
         for s in safras if s not in safras_ref},
        name=x.name, dtype=float)


def psi_por_safra(df, var, col_safra="safra", safras_ref=None, n_bins=10,
                  codigos_especiais=CODIGOS_ESPECIAIS):
    """PSI de cada safra contra uma referência (padrão: as 2 primeiras safras)."""
    return _psi_por_safra(df[var], df[col_safra], safras_ref, n_bins, codigos_especiais)


def classificar_psi(valor):
    if pd.isna(valor):
        return "sem dado"
    if valor < 0.10:
        return "estável"
    if valor < 0.25:
        return "atenção"
    return "instável"


# --------------------------------------------------------------------------- #
# Ficha completa
# --------------------------------------------------------------------------- #
def ficha_variaveis(treino, variaveis, col_target="adimplencia", col_safra="safra",
                    categoricas=(), codigos_especiais=CODIGOS_ESPECIAIS, verbose=True):
    """Calcula IV, PSI e indicadores de qualidade para cada variável.

    Retorna (ficha, binnings):
      ficha: DataFrame com uma linha por variável e as classificações
      binnings: dict var -> OptimalBinning (para ver a tabela de WoE depois)
    """
    linhas, binnings = [], {}
    y = treino[col_target]
    safra = treino[col_safra]
    for i, var in enumerate(variaveis, 1):
        x = treino[var]
        tipo = "categorical" if var in categoricas else "numerical"
        linha = {"variavel": var, "tipo": tipo, "n_unicos": x.nunique(dropna=False),
                 "pct_nan": x.isna().mean()}
        if tipo == "numerical":
            for c in codigos_especiais:
                linha[f"pct_{c}"] = (x == c).mean()
        else:
            x = _como_categoria(x)

        if linha["n_unicos"] <= 1:
            linha.update(iv=0.0, n_faixas=1, psi_max=0.0, psi_medio=0.0)
        else:
            try:
                ob = ajustar_binning(x, y, tipo=tipo, codigos_especiais=codigos_especiais)
                binnings[var] = ob
                linha["iv"] = iv_de(ob)
                linha["n_faixas"] = len(ob.splits) + 1 if tipo == "numerical" else len(ob.splits)
            except Exception as erro:  # registra e segue
                linha.update(iv=np.nan, n_faixas=np.nan, erro=str(erro)[:120])
            psis = _psi_por_safra(x, safra, codigos_especiais=codigos_especiais)
            linha.update(psi_max=psis.max(), psi_medio=psis.mean())
        linhas.append(linha)
        if verbose and i % 25 == 0:
            print(f"{i}/{len(variaveis)} variáveis processadas")

    ficha = pd.DataFrame(linhas).set_index("variavel")
    ficha["classe_iv"] = ficha["iv"].apply(lambda v: classificar_iv(v) if pd.notna(v) else "erro")
    ficha["classe_psi"] = ficha["psi_max"].apply(classificar_psi)
    ficha["sugestao"] = ficha.apply(_sugestao, axis=1)
    return ficha.sort_values("iv", ascending=False), binnings


def _sugestao(l):
    """Sugestão inicial; a decisão final é da equipe, com ablação."""
    if l["n_unicos"] <= 1:
        return "remover (constante)"
    if l["classe_iv"] == "erro":
        return "erro no binning (ver coluna erro)"
    if l["classe_iv"] == "suspeito (revisar)":
        return "revisar vazamento"
    if l["classe_psi"] == "instável":
        return "investigar estabilidade"
    if l["classe_iv"] == "sem poder":
        return "candidata a remoção (testar)"
    if l["classe_psi"] == "atenção":
        return "manter com atenção"
    return "manter"


# --------------------------------------------------------------------------- #
# Correlação entre variáveis
# --------------------------------------------------------------------------- #
# [IA] Funções de correlação geradas com auxílio de IA (Claude) em 11/10/2026.

def correlacao_spearman(df, variaveis):
    """Matriz de correlação de Spearman entre variáveis numéricas.

    Spearman compara a ORDEM dos valores, não os valores em si. Assim os códigos
    especiais (-1, -2, 9999) não distorcem a medida como distorceriam a de
    Pearson: um 9999 conta só como "o maior valor", e não como um número enorme.
    NaN recebe o posto médio da coluna (há pouquíssimos nas VPs).
    """
    postos = df[variaveis].rank().astype("float32")
    postos = postos.fillna(postos.mean())
    matriz = np.corrcoef(postos.to_numpy(), rowvar=False)
    return pd.DataFrame(matriz, index=variaveis, columns=variaveis)


def pares_correlacionados(corr, limiar=0.90):
    """Pares de variáveis com |correlação| >= limiar, do mais forte para o mais fraco."""
    acima = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pares = corr.where(acima).stack().rename("correlacao").reset_index()
    pares.columns = ["var_1", "var_2", "correlacao"]
    pares = pares[pares["correlacao"].abs() >= limiar]
    ordem = pares["correlacao"].abs().sort_values(ascending=False, kind="stable").index
    return pares.loc[ordem].reset_index(drop=True)


def selecionar_por_correlacao(corr, iv, limiar=0.90):
    """Escolhe uma variável em cada grupo de variáveis muito correlacionadas.

    Percorre as variáveis do maior IV para o menor. Cada uma fica se não tiver
    |correlação| >= limiar com nenhuma das que já ficaram; senão sai, e fica
    registrada qual variável (de IV maior) ficou no lugar dela. Em caso de IV
    igual, vale a ordem das colunas.

    Devolve (ficam, saem): a lista das que ficam e um dict {sai: fica_no_lugar}.
    """
    ordem = iv.loc[corr.index].sort_values(ascending=False, kind="stable").index
    ficam, saem = [], {}
    for var in ordem:
        parecida = next((f for f in ficam if abs(corr.loc[var, f]) >= limiar), None)
        if parecida is None:
            ficam.append(var)
        else:
            saem[var] = parecida
    return ficam, saem
