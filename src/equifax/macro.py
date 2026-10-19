"""Dados macroeconômicos: catálogo, coleta (BCB SGS e IBGE SIDRA) e arquivos brutos.

[IA] Módulo gerado com auxílio de IA (Claude) em 11/10/2026. Registrar em docs/uso_ia.md.

Bloco D do spec. Esta parte faz os itens 1 e 2: o catálogo de séries e a coleta.
A junção com a base (sem vazamento) vem no item 3, em `montar_tabela_macro` e
`juntar_macro`.

Coleta (no computador da equipe; a IA não alcança as APIs):

    python -m equifax.macro

Grava cada série em `data/external/macro_bruto/<data do download>/<serie>.csv`
e um `manifesto.json` com URL, data e hora do download e número de linhas.
São dados públicos; nenhuma linha da base do datathon passa por aqui.

Regra de disponibilidade (usada no item 3): a concessão é tratada como feita no
dia 1 do mês da safra. Uma observação com mês de referência r só pode ser usada
a partir do dia 1 do mês r + defasagem_meses (séries diárias: a partir do dia
seguinte ao da observação). As defasagens são conservadoras e estão
justificadas em `docs/macro_catalogo.md`.
"""
import json
import time
import urllib.parse
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_EXTERNAL

MACRO_BRUTO = DATA_EXTERNAL / "macro_bruto"

# Janela de coleta: cobre 12 meses de variação + a maior defasagem antes de
# 202301 e as safras da OOT cega, com folga.
INICIO_PADRAO = date(2020, 1, 1)
FIM_PADRAO = date(2024, 12, 31)

URL_SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
URL_SIDRA = "https://apisidra.ibge.gov.br/values/t/{tabela}/{nivel}/all/v/{variavel}/p/all"


@dataclass(frozen=True)
class Serie:
    nome: str
    conceito: str
    fonte: str              # "sgs" ou "sidra"
    codigo: str             # código SGS ou tabela SIDRA
    unidade: str
    frequencia: str         # "diaria", "mensal", "trimestre_movel" ou "trimestral"
    geografia: str          # "BR" ou "UF"
    defasagem_meses: int    # 0 para diárias (vale a partir do dia seguinte)
    revisada: bool
    hipotese: str
    variavel_sidra: str = ""
    nivel_sidra: str = ""   # "n1" (Brasil) ou "n3" (UF)


CATALOGO = {s.nome: s for s in [
    Serie("selic_meta", "Meta da taxa Selic definida pelo Copom", "sgs", "432", "% a.a.",
          "diaria", "BR", 0, False,
          "Juro alto encarece o crédito e aperta o orçamento das famílias"),
    Serie("ipca_mes", "IPCA, variação mensal", "sgs", "433", "% no mês",
          "mensal", "BR", 2, False, "Inflação reduz a renda real"),
    Serie("ipca_12m", "IPCA, variação acumulada em 12 meses", "sgs", "13522", "% em 12 meses",
          "mensal", "BR", 2, False, "Inflação reduz a renda real"),
    Serie("inadimplencia_pf", "Inadimplência da carteira de crédito, pessoas físicas, total",
          "sgs", "21084", "%", "mensal", "BR", 3, True,
          "Retrato direto do ciclo de crédito"),
    Serie("comprometimento_renda",
          "Comprometimento de renda das famílias com o serviço da dívida com o SFN, "
          "com ajuste sazonal", "sgs", "29034", "%", "mensal", "BR", 4, True,
          "Famílias com mais renda comprometida pagam pior"),
    Serie("endividamento_familias",
          "Endividamento das famílias com o SFN em relação à renda acumulada dos "
          "últimos 12 meses", "sgs", "29037", "%", "mensal", "BR", 4, True,
          "Famílias já endividadas pagam pior"),
    Serie("desocupacao_br", "Taxa de desocupação (PNAD Contínua), trimestre móvel; "
          "referência = último mês do trimestre", "sidra", "6381", "%",
          "trimestre_movel", "BR", 3, False,
          "Desemprego derruba a capacidade de pagar", variavel_sidra="4099", nivel_sidra="n1"),
    Serie("desocupacao_uf", "Taxa de desocupação (PNAD Contínua), trimestral, por UF; "
          "referência = último mês do trimestre", "sidra", "4099", "%",
          "trimestral", "UF", 4, False,
          "Desemprego derruba a capacidade de pagar; varia entre clientes da mesma safra",
          variavel_sidra="4099", nivel_sidra="n3"),
]}

COLUNAS_BRUTO = ["serie", "periodo_codigo", "data_referencia", "territorio_codigo",
                 "territorio", "valor"]


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _baixar_json(url, tentativas=3, espera=5, timeout=60):
    """GET com algumas tentativas; devolve o JSON decodificado."""
    pedido = urllib.request.Request(url, headers={"User-Agent": "equifax-datathon/0.1",
                                                  "Accept": "application/json"})
    ultimo_erro = None
    for tentativa in range(1, tentativas + 1):
        try:
            with urllib.request.urlopen(pedido, timeout=timeout) as resposta:
                return json.loads(resposta.read().decode("utf-8"))
        except Exception as erro:  # rede, HTTP ou JSON inválido
            ultimo_erro = erro
            if tentativa < tentativas:
                time.sleep(espera * tentativa)
    raise RuntimeError(f"Falha ao baixar {url}: {ultimo_erro}")


def url_sgs(codigo, inicio=INICIO_PADRAO, fim=FIM_PADRAO):
    consulta = urllib.parse.urlencode({"formato": "json",
                                       "dataInicial": inicio.strftime("%d/%m/%Y"),
                                       "dataFinal": fim.strftime("%d/%m/%Y")})
    return URL_SGS.format(codigo=codigo) + "?" + consulta


def url_sidra(serie):
    return URL_SIDRA.format(tabela=serie.codigo, nivel=serie.nivel_sidra,
                            variavel=serie.variavel_sidra)


# --------------------------------------------------------------------------- #
# Leitura das respostas (separada do download para ser testada sem rede)
# --------------------------------------------------------------------------- #
def ler_resposta_sgs(dados, serie):
    """Lista [{'data': 'dd/mm/aaaa', 'valor': '13.75'}, ...] -> tabela bruta."""
    if not isinstance(dados, list):
        raise ValueError(f"Resposta inesperada do SGS para {serie.nome}: {str(dados)[:200]}")
    df = pd.DataFrame(dados, columns=["data", "valor"])
    datas = pd.to_datetime(df["data"], format="%d/%m/%Y")
    return pd.DataFrame({
        "serie": serie.nome,
        "periodo_codigo": df["data"],
        "data_referencia": datas.dt.date if serie.frequencia == "diaria"
        else datas.dt.to_period("M").dt.to_timestamp().dt.date,
        "territorio_codigo": "BR",
        "territorio": "Brasil",
        "valor": pd.to_numeric(df["valor"], errors="coerce"),
    }, columns=COLUNAS_BRUTO)


def _mes_de_referencia(codigo, frequencia):
    """Período do SIDRA -> primeiro dia do último mês do período."""
    ano, resto = int(codigo[:4]), int(codigo[4:])
    if frequencia == "trimestral":       # AAAATT, TT = 01..04
        if not 1 <= resto <= 4:
            raise ValueError(f"Trimestre inválido: {codigo}")
        return date(ano, 3 * resto, 1)
    if frequencia == "trimestre_movel":  # AAAAMM = último mês do trimestre
        return date(ano, resto, 1)
    raise ValueError(f"Frequência sem regra para o SIDRA: {frequencia}")


def ler_resposta_sidra(dados, serie):
    """Lista do SIDRA (1ª linha = cabeçalho) -> tabela bruta.

    Usa as colunas V (valor), D1C/D1N (território) e a dimensão do período,
    achada pelo nome no cabeçalho (Trimestre, Trimestre Móvel, Mês).
    """
    if not isinstance(dados, list) or len(dados) < 1:
        raise ValueError(f"Resposta inesperada do SIDRA para {serie.nome}: {str(dados)[:200]}")
    cabecalho, linhas = dados[0], dados[1:]
    chave_periodo = next((k for k, v in cabecalho.items()
                          if k.startswith("D") and k.endswith("C")
                          and str(v).split(" (")[0] in ("Trimestre", "Trimestre Móvel", "Mês")),
                         None)
    if chave_periodo is None:
        raise ValueError(f"Período não achado no cabeçalho do SIDRA: {cabecalho}")
    df = pd.DataFrame(linhas)
    return pd.DataFrame({
        "serie": serie.nome,
        "periodo_codigo": df[chave_periodo].astype(str),
        "data_referencia": [_mes_de_referencia(c, serie.frequencia) for c in df[chave_periodo]],
        "territorio_codigo": df["D1C"].astype(str).replace({"1": "BR"}),
        "territorio": df["D1N"],
        # o SIDRA usa "-", "..." ou "X" para valor ausente ou sigiloso
        "valor": pd.to_numeric(df["V"], errors="coerce"),
    }, columns=COLUNAS_BRUTO)


def coletar_sgs(serie, inicio=INICIO_PADRAO, fim=FIM_PADRAO, baixar=_baixar_json):
    url = url_sgs(serie.codigo, inicio, fim)
    return ler_resposta_sgs(baixar(url), serie), url


def coletar_sidra(serie, inicio=INICIO_PADRAO, fim=FIM_PADRAO, baixar=_baixar_json):
    url = url_sidra(serie)
    bruto = ler_resposta_sidra(baixar(url), serie)
    noperiodo = (bruto["data_referencia"] >= inicio) & (bruto["data_referencia"] <= fim)
    return bruto[noperiodo].reset_index(drop=True), url


def coletar(serie, inicio=INICIO_PADRAO, fim=FIM_PADRAO, baixar=_baixar_json):
    """Baixa uma série do catálogo. Devolve (tabela bruta, URL usada)."""
    if serie.fonte == "sgs":
        return coletar_sgs(serie, inicio, fim, baixar)
    if serie.fonte == "sidra":
        return coletar_sidra(serie, inicio, fim, baixar)
    raise ValueError(f"Fonte desconhecida: {serie.fonte}")


# --------------------------------------------------------------------------- #
# Coleta de todo o catálogo
# --------------------------------------------------------------------------- #
def baixar_catalogo(pasta=MACRO_BRUTO, series=None, inicio=INICIO_PADRAO, fim=FIM_PADRAO,
                    baixar=_baixar_json, agora=None):
    """Baixa as séries e grava em <pasta>/<AAAA-MM-DD>/, com um manifesto.

    Uma série que falha não interrompe as outras: o erro vai para o manifesto.
    Devolve (pasta do download, manifesto).
    """
    agora = agora or datetime.now()
    destino = Path(pasta) / agora.strftime("%Y-%m-%d")
    destino.mkdir(parents=True, exist_ok=True)
    nomes = list(series) if series is not None else list(CATALOGO)

    manifesto = {"baixado_em": agora.isoformat(timespec="seconds"),
                 "inicio": inicio.isoformat(), "fim": fim.isoformat(), "series": {}}
    for nome in nomes:
        serie = CATALOGO[nome]
        registro = {"catalogo": asdict(serie)}
        try:
            bruto, url = coletar(serie, inicio, fim, baixar)
            if bruto.empty:
                raise ValueError("a série veio vazia")
            bruto.to_csv(destino / f"{nome}.csv", index=False, lineterminator="\n")
            registro.update(url=url, arquivo=f"{nome}.csv", n_linhas=len(bruto),
                            n_territorios=int(bruto["territorio_codigo"].nunique()),
                            primeira_referencia=str(bruto["data_referencia"].min()),
                            ultima_referencia=str(bruto["data_referencia"].max()),
                            valores_ausentes=int(bruto["valor"].isna().sum()))
        except Exception as erro:
            registro["erro"] = str(erro)[:500]
        manifesto["series"][nome] = registro

    with open(destino / "manifesto.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifesto, f, ensure_ascii=False, indent=2)
    return destino, manifesto


def ler_bruto(pasta):
    """Junta os CSVs brutos de um download numa tabela só."""
    pasta = Path(pasta)
    partes = [pd.read_csv(arq, dtype={"periodo_codigo": str, "territorio_codigo": str},
                          parse_dates=["data_referencia"])
              for arq in sorted(pasta.glob("*.csv"))]
    if not partes:
        raise FileNotFoundError(f"Nenhum CSV em {pasta}")
    return pd.concat(partes, ignore_index=True)


# Valores conhecidos, publicados nos comunicados oficiais, para conferir se cada
# código baixou a série certa (mês de referência -> valor, tolerância).
CONFERENCIA = {
    "selic_meta": ("2023-03-01", 13.75, 0.001),     # meta de 13,75% de ago/22 a ago/23
    "ipca_12m": ("2022-12-01", 5.79, 0.05),         # IPCA de 2022: 5,79%
    "ipca_mes": ("2023-01-01", 0.53, 0.02),         # IPCA de jan/23: 0,53%
}


def resumo(pasta):
    """Tabela curta por série (para mandar de volta à IA: são dados públicos)."""
    bruto = ler_bruto(pasta)
    linhas = []
    for nome, g in bruto.groupby("serie", sort=False):
        linha = {"serie": nome, "linhas": len(g), "territorios": g["territorio_codigo"].nunique(),
                 "primeira": g["data_referencia"].min().date(),
                 "ultima": g["data_referencia"].max().date(),
                 "min": g["valor"].min(), "max": g["valor"].max(),
                 "ausentes": int(g["valor"].isna().sum()), "conferencia": ""}
        if nome in CONFERENCIA:
            ref, esperado, tol = CONFERENCIA[nome]
            v = g.loc[g["data_referencia"] == pd.Timestamp(ref), "valor"]
            if nome == "selic_meta":
                v = g.loc[g["data_referencia"].dt.to_period("M") == pd.Period(ref, "M"), "valor"]
            ok = len(v) and abs(float(v.iloc[-1]) - esperado) <= tol
            linha["conferencia"] = (f"ok ({float(v.iloc[-1])} em {ref[:7]})" if ok else
                                    f"CONFERIR: esperado {esperado} em {ref[:7]}, veio "
                                    f"{list(v.round(3))[-1:] or 'nada'}")
        linhas.append(linha)
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
# Tabela macro com datas de disponibilidade (bloco D, item 3)
# --------------------------------------------------------------------------- #
SEM_INFORMACAO = "Sem informação"

# Séries por UF e a série nacional usada quando o estado do cliente é desconhecido.
# A desocupação trimestral por UF (fim de trimestre em março) e o trimestre móvel
# nacional terminado no mesmo mês (jan-fev-mar) cobrem o mesmo período.
# Para a diferença da UF para o Brasil, o substituto é a constante 0 (o próprio Brasil).
SUBSTITUTA_NACIONAL = {"desocupacao_uf": "desocupacao_br",
                       "desocupacao_uf_var_12m": "desocupacao_br_var_12m",
                       "desocupacao_uf_menos_br": 0.0}

COLUNAS_TABELA = ["serie", "data_referencia", "uf", "valor", "disponivel_em"]


def pasta_mais_recente(pasta=MACRO_BRUTO):
    """Pasta do download mais recente em data/external/macro_bruto/."""
    datadas = sorted(p for p in Path(pasta).iterdir() if p.is_dir() and p.name[:4].isdigit())
    if not datadas:
        raise FileNotFoundError(f"Nenhum download em {pasta}; rode python -m equifax.macro")
    return datadas[-1]


def disponivel_em(data_referencia, serie):
    """Primeiro dia em que a observação pode ser usada.

    Diária: o dia seguinte. Demais: o dia 1 do mês de referência + defasagem.
    """
    ref = pd.to_datetime(pd.Series(data_referencia))
    if serie.frequencia == "diaria":
        return (ref + pd.Timedelta(days=1)).to_numpy()
    meses = ref.dt.to_period("M") + serie.defasagem_meses
    return meses.dt.to_timestamp().to_numpy()


def _variacao(tabela, nome, meses, novo_nome):
    """Diferença para o valor de `meses` antes (na mesma UF), em pontos.

    O valor antigo é anterior ao atual, então já estava disponível quando o atual
    ficou disponível: a data de disponibilidade é a do valor atual.
    """
    t = tabela[tabela["serie"] == nome].copy()
    anterior = t[["uf", "data_referencia", "valor"]].copy()
    anterior["data_referencia"] = anterior["data_referencia"] + pd.DateOffset(months=meses)
    t = t.merge(anterior.rename(columns={"valor": "valor_antes"}),
                on=["uf", "data_referencia"], how="inner")
    t["valor"] = t["valor"] - t["valor_antes"]
    t["serie"] = novo_nome
    return t[COLUNAS_TABELA]


def _uf_menos_nacional(tabela, nome_uf, nome_br, novo_nome):
    """Valor da UF menos o nacional do mesmo mês de referência; vale quando os dois valem."""
    uf = tabela[tabela["serie"] == nome_uf]
    br = tabela[tabela["serie"] == nome_br][["data_referencia", "valor", "disponivel_em"]]
    t = uf.merge(br, on="data_referencia", suffixes=("", "_br"), how="inner")
    t["valor"] = t["valor"] - t["valor_br"]
    t["disponivel_em"] = t[["disponivel_em", "disponivel_em_br"]].max(axis=1)
    t["serie"] = novo_nome
    return t[COLUNAS_TABELA]


TRANSFORMACOES = {
    # nome novo: (função, argumentos)
    "ipca_12m_var_3m": (_variacao, ("ipca_12m", 3)),
    "inadimplencia_pf_var_3m": (_variacao, ("inadimplencia_pf", 3)),
    "inadimplencia_pf_var_12m": (_variacao, ("inadimplencia_pf", 12)),
    "comprometimento_renda_var_12m": (_variacao, ("comprometimento_renda", 12)),
    "endividamento_familias_var_12m": (_variacao, ("endividamento_familias", 12)),
    "desocupacao_br_var_12m": (_variacao, ("desocupacao_br", 12)),
    "desocupacao_uf_var_12m": (_variacao, ("desocupacao_uf", 12)),
    "desocupacao_uf_menos_br": (_uf_menos_nacional, ("desocupacao_uf", "desocupacao_br")),
}


def montar_tabela_macro(pasta=None, transformacoes=True):
    """Tabela longa: série, mês (ou dia) de referência, UF, valor e data de disponibilidade.

    uf = "BR" nas séries nacionais e o nome do estado (igual à coluna `estado`
    da base) nas séries por UF. Com transformacoes=True, acrescenta as séries de
    TRANSFORMACOES (variações e diferença da UF para o Brasil).
    """
    bruto = ler_bruto(pasta or pasta_mais_recente())
    partes = []
    for nome, g in bruto.groupby("serie", sort=False):
        if nome not in CATALOGO:
            raise ValueError(f"Série fora do catálogo: {nome}")
        serie = CATALOGO[nome]
        t = pd.DataFrame({
            "serie": nome,
            "data_referencia": pd.to_datetime(g["data_referencia"]).to_numpy(),
            "uf": g["territorio"].to_numpy() if serie.geografia == "UF" else "BR",
            "valor": g["valor"].to_numpy(),
        })
        t["disponivel_em"] = disponivel_em(t["data_referencia"], serie)
        partes.append(t)
    tabela = pd.concat(partes, ignore_index=True)
    tabela = tabela.dropna(subset=["valor"])
    if transformacoes:
        presentes = set(tabela["serie"])
        novas = [f(tabela, *args, novo) for novo, (f, args) in TRANSFORMACOES.items()
                 if all(a in presentes for a in args if isinstance(a, str))]
        tabela = pd.concat([tabela, *novas], ignore_index=True)
    return tabela.sort_values(["serie", "uf", "data_referencia"]).reset_index(drop=True)[COLUNAS_TABELA]


# --------------------------------------------------------------------------- #
# Junção com a base, sem vazamento (bloco D, item 3)
# --------------------------------------------------------------------------- #
def data_da_concessao(safra):
    """Dia 1 do mês da safra (convenção do projeto: a hipótese mais conservadora)."""
    return pd.to_datetime(pd.Series(safra).astype(int).astype(str), format="%Y%m")


def _ultimo_disponivel(tabela_serie, datas):
    """Para cada data, a observação mais recente (por referência) já disponível nela.

    Devolve DataFrame indexado pela data, com `valor` (float) e `data_referencia`
    (NaN e NaT quando nada estava disponível).
    """
    t = tabela_serie.sort_values("data_referencia")
    disp = pd.to_datetime(t["disponivel_em"]).to_numpy()
    ref = pd.to_datetime(t["data_referencia"]).to_numpy()
    val = t["valor"].to_numpy(dtype=float)
    valores, referencias = [], []
    for d in datas:
        ok = np.flatnonzero(disp <= np.datetime64(d))
        if len(ok):
            k = ok[-1]  # ordenado por referência: a última disponível é a mais recente
            valores.append(val[k])
            referencias.append(ref[k])
        else:
            valores.append(np.nan)
            referencias.append(np.datetime64("NaT", "ns"))
    indice = pd.DatetimeIndex(datas)
    return pd.DataFrame({"valor": np.array(valores, dtype=float),
                         "data_referencia": pd.to_datetime(referencias)}, index=indice)


def _idade_meses(data_concessao, data_referencia):
    con = pd.to_datetime(data_concessao).dt.to_period("M")
    ref = pd.to_datetime(data_referencia).dt.to_period("M")
    idade = pd.Series(np.nan, index=con.index)
    ok = ref.notna()
    idade[ok] = [(c - r).n for c, r in zip(con[ok], ref[ok])]
    return idade


def juntar_macro(base, tabela, series=None, col_safra="safra", col_estado="estado",
                 prefixo="macro_", idade=True):
    """Colunas macro de cada linha de `base`, com o mesmo índice e a mesma ordem.

    Para cada linha, cada série recebe o último valor já disponível no dia 1 do
    mês da safra (`disponivel_em <= data da concessão`). Séries por UF usam o
    `estado` do cliente; "Sem informação" recebe a série nacional de
    SUBSTITUTA_NACIONAL. Com idade=True, cada série original ganha a coluna
    <serie>_idade_meses (meses entre a referência usada e a concessão).
    Não altera `base`.
    """
    series = list(series) if series is not None else list(dict.fromkeys(tabela["serie"]))
    faltando = sorted(set(series) - set(tabela["serie"]))
    if faltando:
        raise KeyError(f"Séries que não estão na tabela: {faltando}")

    concessao = data_da_concessao(base[col_safra].to_numpy())
    concessao.index = base.index
    datas = sorted(concessao.unique())
    saida = pd.DataFrame(index=base.index)

    estados = None
    for nome in series:
        t = tabela[tabela["serie"] == nome]
        por_uf = (t["uf"] != "BR").any()
        if not por_uf:
            ult = _ultimo_disponivel(t, datas)
            valor = concessao.map(ult["valor"]).astype(float)
            ref = concessao.map(ult["data_referencia"])
        else:
            if estados is None:
                estados = base[col_estado].astype(object).where(base[col_estado].notna(),
                                                                SEM_INFORMACAO).astype(str)
                desconhecidos = sorted(set(estados) - set(t["uf"]) - {SEM_INFORMACAO})
                if desconhecidos:
                    raise ValueError(f"Estados sem valor na série {nome}: {desconhecidos}")
            valor = pd.Series(np.nan, index=base.index)
            ref = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns]")
            for uf, tu in t.groupby("uf"):
                linhas = estados == uf
                if linhas.any():
                    ult = _ultimo_disponivel(tu, datas)
                    valor[linhas] = concessao[linhas].map(ult["valor"]).astype(float)
                    ref[linhas] = concessao[linhas].map(ult["data_referencia"])
            substituta = SUBSTITUTA_NACIONAL.get(nome)
            sem_info = estados == SEM_INFORMACAO
            if sem_info.any() and isinstance(substituta, (int, float)):
                valor[sem_info] = float(substituta)
            elif sem_info.any() and substituta:
                tb = tabela[tabela["serie"] == substituta]
                ult = _ultimo_disponivel(tb, datas)
                valor[sem_info] = concessao[sem_info].map(ult["valor"]).astype(float)
                ref[sem_info] = concessao[sem_info].map(ult["data_referencia"])
        saida[prefixo + nome] = valor.to_numpy()
        if idade and nome in CATALOGO:
            saida[prefixo + nome + "_idade_meses"] = _idade_meses(concessao, ref).to_numpy()

    vazias = [c for c in saida.columns if saida[c].isna().any()]
    if vazias:
        warnings.warn(f"Colunas macro com valores ausentes (sem dado disponível na data): {vazias}")
    return saida


class JuntarMacro:
    """Junção macro guardada no pacote do modelo (escoragem).

    A tabela vai junto no pacote, para a OOT receber exatamente os valores usados no
    desenvolvimento. Em produção, a tabela seria montada do download mais recente.
    As colunas de idade do dado entram na base (não no modelo) para o relatório.
    """

    def __init__(self, tabela, series):
        self.tabela = tabela
        self.series = list(series)

    def __call__(self, base):
        colunas = juntar_macro(base, self.tabela, series=self.series)
        return pd.concat([base, colunas], axis=1)


def adicionar_macro(base, tabela, **kwargs):
    """Acrescenta as colunas de `juntar_macro` à própria base (sem cópia) e a devolve."""
    colunas = juntar_macro(base, tabela, **kwargs)
    for c in colunas.columns:
        base[c] = colunas[c]
    return base


if __name__ == "__main__":
    destino, manifesto = baixar_catalogo()
    print(f"Download em {destino}")
    for nome, reg in manifesto["series"].items():
        print(f"  {nome:24} " + (f"ERRO: {reg['erro']}" if "erro" in reg
                                 else f"{reg['n_linhas']} linhas"))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(resumo(destino).to_string(index=False))
