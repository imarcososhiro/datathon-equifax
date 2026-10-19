"""Feature engineering do M2 (bloco C): famílias de variáveis novas.

[IA] Módulo gerado com auxílio de IA (Claude) em 13/10/2026. Registrar em docs/uso_ia.md.

Cada família é uma classe com:
- `ajustar(treino)`: aprende o que precisar SÓ do treino (medianas, taxas);
- `aplicar(base)`: devolve um DataFrame com as colunas novas, mesmo índice da base;
- `ajustar_aplicar(treino)`: as colunas do próprio treino. Para quem usa o target
  (taxa de inadimplência por prefixo de CEP), as linhas do treino recebem valores
  calculados fora da dobra, para o modelo não ver a própria resposta.

Assim a OOT recebe exatamente a transformação aprendida no treino. Todas as
colunas novas começam com `fe_`. Nenhum nome de VP fica escrito aqui: a lista de
VPs usada nas contagens vem de fora.
"""
import re

import numpy as np
import pandas as pd

from .config import RANDOM_SEED
from .tratamento import ORDEM_RENDAS, SEM_INFORMACAO

CODIGOS = {"menos1": -1, "menos2": -2, "9999": 9999, "zero": 0}

# Regiões do IBGE: só entram aqui, como hipótese de feature engineering (famílias 2 e 3 e a
# troca de `estado` por `regiao` no 04). A base tratada e o Modelo Base usam o `estado`.
# [IA] - Utilizamos IA para montar o dicionário REGIOES, que mapeia cada estado para a
# sua região do IBGE. O laço que monta o ESTADO_PARA_REGIAO foi reescrito pela equipe.
REGIOES = {
    "Norte": ["Acre", "Amapá", "Amazonas", "Pará", "Rondônia", "Roraima", "Tocantins"],
    "Nordeste": ["Alagoas", "Bahia", "Ceará", "Maranhão", "Paraíba", "Pernambuco", "Piauí",
                 "Rio Grande do Norte", "Sergipe"],
    "Centro-Oeste": ["Distrito Federal", "Goiás", "Mato Grosso", "Mato Grosso do Sul"],
    "Sudeste": ["Espírito Santo", "Minas Gerais", "Rio de Janeiro", "São Paulo"],
    "Sul": ["Paraná", "Rio Grande do Sul", "Santa Catarina"],
    SEM_INFORMACAO: [SEM_INFORMACAO],
}
ESTADO_PARA_REGIAO = {}

for _regiao, _estados in REGIOES.items():
    for _estado in _estados:
        ESTADO_PARA_REGIAO[_estado] = _regiao


def _texto(serie):
    return serie.astype(object).where(serie.notna(), SEM_INFORMACAO).astype(str)


def regiao_do_estado(estado):
    """Região de cada estado; estado fora do dicionário vira 'Sem informação'."""
    return _texto(estado).map(ESTADO_PARA_REGIAO).fillna(SEM_INFORMACAO)


def _renda_ordinal(serie):
    """0 = '3 - 1000' ... 9 = '25001 - 35000'; fora da lista vira NaN."""
    return _texto(serie).map({r: i for i, r in enumerate(ORDEM_RENDAS)}).astype(float)


def _idade_ordinal(serie):
    """Ordem das faixas de idade pelo primeiro número ('18 - 25' -> 18); 'Sem informação' = NaN."""
    def inicio(rotulo):
        m = re.match(r"\s*(\d+)", rotulo)
        return float(m.group(1)) if m else np.nan
    texto = _texto(serie)
    inicios = {r: inicio(r) for r in texto.unique()}
    ordem = {v: i for i, v in enumerate(sorted(x for x in set(inicios.values()) if not np.isnan(x)))}
    return texto.map(lambda r: ordem.get(inicios[r], np.nan)).astype(float)


class Familia:
    nome = ""
    hipotese = ""

    def ajustar(self, treino):
        return self

    def aplicar(self, base):
        raise NotImplementedError

    def ajustar_aplicar(self, treino):
        return self.ajustar(treino).aplicar(treino)


# --------------------------------------------------------------------------- #
# Família 1: contagens por linha
# --------------------------------------------------------------------------- #
class ContagensPorLinha(Familia):
    nome = "f1_contagens"
    hipotese = ("Quantas VPs têm cada código especial (-1, -2, 9999), zero ou nulo: a quantidade "
                "de códigos por linha pode carregar informação de risco (\"o evento nunca "
                "aconteceu\").")

    def __init__(self, vps):
        self.vps = list(vps)

    def aplicar(self, base):
        valores = base[self.vps].to_numpy(dtype="float32")
        saida = {f"fe_n_{nome}": (valores == codigo).sum(axis=1) for nome, codigo in CODIGOS.items()}
        saida["fe_n_nulo"] = np.isnan(valores).sum(axis=1)
        return pd.DataFrame(saida, index=base.index).astype("float32")


# --------------------------------------------------------------------------- #
# Família 2: renda relativa à UF e à região
# --------------------------------------------------------------------------- #
class RendaRelativa(Familia):
    nome = "f2_renda_relativa"
    hipotese = ("A mesma renda vale diferente em cada lugar (exemplo do próprio regulamento): "
                "estar acima ou abaixo da renda típica do estado ou da região diz algo além da "
                "faixa de renda sozinha.")

    def ajustar(self, treino):
        renda = _renda_ordinal(treino["faixa_renda"])
        uf = _texto(treino["estado"])
        reg = regiao_do_estado(uf)
        self.mediana_geral_ = float(renda.median())
        self.mediana_uf_ = renda.groupby(uf).median().to_dict()
        self.mediana_regiao_ = renda.groupby(reg).median().to_dict()
        # % de clientes do estado com faixa de renda menor ou igual (CDF empírica)
        self.cdf_uf_ = {}
        for nome, r in renda.groupby(uf):
            contagem = r.value_counts(normalize=True).sort_index().cumsum()
            self.cdf_uf_[nome] = contagem.to_dict()
        contagem = renda.value_counts(normalize=True).sort_index().cumsum()
        self.cdf_geral_ = contagem.to_dict()
        return self

    def aplicar(self, base):
        renda = _renda_ordinal(base["faixa_renda"])
        uf = _texto(base["estado"])
        reg = regiao_do_estado(uf)
        med_uf = uf.map(self.mediana_uf_).fillna(self.mediana_geral_)
        med_reg = reg.map(self.mediana_regiao_).fillna(self.mediana_geral_)
        pct = [self.cdf_uf_.get(u, self.cdf_geral_).get(r, np.nan) if not np.isnan(r) else np.nan
               for u, r in zip(uf, renda)]
        return pd.DataFrame({"fe_renda_ord": renda,
                             "fe_renda_menos_mediana_uf": renda - med_uf,
                             "fe_renda_menos_mediana_regiao": renda - med_reg,
                             "fe_renda_pct_uf": pct}, index=base.index).astype("float32")


# --------------------------------------------------------------------------- #
# Família 3: interações demográficas
# --------------------------------------------------------------------------- #
class Interacoes(Familia):
    nome = "f3_interacoes"
    hipotese = ("A idade pode ter efeito próprio dentro de cada faixa de renda: a combinação "
                "renda x idade, e idade x região, separa perfis que cada variável sozinha mistura.")

    def __init__(self, minimo=200):
        self.minimo = minimo  # combinações com menos clientes no treino viram "outros"

    def _combinacoes(self, base):
        renda = _texto(base["faixa_renda"])
        idade = _texto(base["faixa_idade"])
        reg = regiao_do_estado(base["estado"])
        return renda + " | " + idade, idade + " | " + reg

    def ajustar(self, treino):
        ri, ir = self._combinacoes(treino)
        self.validas_ri_ = sorted(set(ri.value_counts()[lambda s: s >= self.minimo].index))
        self.validas_ir_ = sorted(set(ir.value_counts()[lambda s: s >= self.minimo].index))
        return self

    def aplicar(self, base):
        ri, ir = self._combinacoes(base)
        cat_ri = self.validas_ri_ + ["outros"]
        cat_ir = self.validas_ir_ + ["outros"]
        renda = _renda_ordinal(base["faixa_renda"])
        idade = _idade_ordinal(base["faixa_idade"])
        return pd.DataFrame({
            "fe_idade_ord": idade.astype("float32"),
            "fe_renda_x_idade": pd.Categorical(ri.where(ri.isin(self.validas_ri_), "outros"), cat_ri),
            "fe_idade_x_regiao": pd.Categorical(ir.where(ir.isin(self.validas_ir_), "outros"), cat_ir),
            "fe_renda_menos_idade": (renda - idade).astype("float32"),
        }, index=base.index)


# --------------------------------------------------------------------------- #
# Família 4: CEP
# --------------------------------------------------------------------------- #
# [IA] Faixas de CEP por UF (Correios), pelos 5 primeiros dígitos. Conferidas em 13/10/2026 com
# https://www.ruacep.com.br/guias/postais/faixa-de-cep-por-estado/ (Roraima fica dentro da faixa do
# Amazonas; DF e Goiás se intercalam; Rondônia está na região 7).
FAIXAS_CEP = [
    (1000, 19999, "São Paulo"), (20000, 28999, "Rio de Janeiro"), (29000, 29999, "Espírito Santo"),
    (30000, 39999, "Minas Gerais"), (40000, 48999, "Bahia"), (49000, 49999, "Sergipe"),
    (50000, 56999, "Pernambuco"), (57000, 57999, "Alagoas"), (58000, 58999, "Paraíba"),
    (59000, 59999, "Rio Grande do Norte"), (60000, 63999, "Ceará"), (64000, 64999, "Piauí"),
    (65000, 65999, "Maranhão"), (66000, 68899, "Pará"), (68900, 68999, "Amapá"),
    (69000, 69299, "Amazonas"), (69300, 69399, "Roraima"), (69400, 69899, "Amazonas"),
    (69900, 69999, "Acre"), (70000, 72799, "Distrito Federal"), (72800, 72999, "Goiás"),
    (73000, 73699, "Distrito Federal"), (73700, 76799, "Goiás"), (76800, 76999, "Rondônia"),
    (77000, 77999, "Tocantins"), (78000, 78899, "Mato Grosso"), (79000, 79999, "Mato Grosso do Sul"),
    (80000, 87999, "Paraná"), (88000, 89999, "Santa Catarina"), (90000, 99999, "Rio Grande do Sul"),
]


def uf_do_cep(cep):
    """UF pelos 5 primeiros dígitos do CEP; NaN se o CEP não for válido."""
    texto = _texto(cep)
    valido = texto.str.fullmatch(r"\d{8}")
    prefixo = pd.to_numeric(texto.where(valido).str[:5], errors="coerce")
    uf = pd.Series(np.nan, index=cep.index, dtype=object)
    for ini, fim, nome in FAIXAS_CEP:
        uf[(prefixo >= ini) & (prefixo <= fim)] = nome
    return uf


class Cep(Familia):
    nome = "f4_cep"
    hipotese = ("Localização mais fina que o estado (prefixo de 3 dígitos), e sinais de cadastro "
                "fraco: CEP ausente, CEP genérico (final 000) e CEP de outro estado.")

    def __init__(self, suavizacao=200, n_dobras=4, semente=RANDOM_SEED):
        self.suavizacao = suavizacao  # peso da média geral na taxa por prefixo
        self.n_dobras = n_dobras
        self.semente = semente

    @staticmethod
    def _prefixo(base):
        texto = _texto(base["cep8"])
        return texto.where(texto.str.fullmatch(r"\d{8}"), SEM_INFORMACAO).str[:3]

    def _taxas(self, prefixo, mau):
        g = mau.groupby(prefixo).agg(["sum", "count"])
        media = float(mau.mean())
        return ((g["sum"] + self.suavizacao * media) / (g["count"] + self.suavizacao)).to_dict(), media

    def ajustar(self, treino):
        mau = 1 - treino["adimplencia"].astype(int)
        self.taxa_prefixo_, self.media_ = self._taxas(self._prefixo(treino), mau)
        return self

    def _sem_target(self, base):
        texto = _texto(base["cep8"])
        valido = texto.str.fullmatch(r"\d{8}")
        uf_cep = uf_do_cep(base["cep8"])
        estado = _texto(base["estado"])
        coerente = np.where(valido & uf_cep.notna() & (estado != SEM_INFORMACAO),
                            (uf_cep == estado).astype(float), np.nan)
        return pd.DataFrame({
            "fe_cep_ausente": (~valido).astype("float32"),
            "fe_cep_generico": (valido & texto.str.endswith("000")).astype("float32"),
            "fe_cep_coerente_uf": coerente.astype("float32"),
        }, index=base.index)

    def aplicar(self, base):
        saida = self._sem_target(base)
        saida["fe_cep3_taxa_inad"] = (self._prefixo(base).map(self.taxa_prefixo_)
                                      .fillna(self.media_).astype("float32"))
        return saida

    def ajustar_aplicar(self, treino):
        """No treino, a taxa de cada linha vem das outras dobras (agrupadas por id)."""
        from .ajuste import dobras_por_id
        self.ajustar(treino)
        saida = self._sem_target(treino)
        prefixo = self._prefixo(treino)
        mau = 1 - treino["adimplencia"].astype(int)
        fora = np.full(len(treino), np.nan, dtype="float64")
        for posicoes_tr, posicoes_va in dobras_por_id(treino, self.n_dobras, self.semente):
            taxas, media = self._taxas(prefixo.iloc[posicoes_tr], mau.iloc[posicoes_tr])
            fora[posicoes_va] = prefixo.iloc[posicoes_va].map(taxas).fillna(media).to_numpy()
        saida["fe_cep3_taxa_inad"] = fora.astype("float32")
        return saida


# --------------------------------------------------------------------------- #
# Aplicar famílias às partes (treino, teste, ...)
# --------------------------------------------------------------------------- #
def gerar_features(partes, familias):
    """Ajusta cada família no treino e aplica em todas as partes.

    Devolve (partes com as colunas novas, dict nome da família -> colunas novas).
    As partes originais não são alteradas.
    """
    novas = {"treino": [], **{k: [] for k in partes if k != "treino"}}
    colunas = {}
    for familia in familias:
        f_treino = familia.ajustar_aplicar(partes["treino"])
        novas["treino"].append(f_treino)
        for k in partes:
            if k != "treino":
                novas[k].append(familia.aplicar(partes[k]))
        colunas[familia.nome] = list(f_treino.columns)
    saida = {k: pd.concat([partes[k], *novas[k]], axis=1) for k in partes}
    return saida, colunas


class AplicarFamilias:
    """Famílias já ajustadas, aplicadas a uma base nova (vai no pacote da escoragem)."""

    def __init__(self, familias):
        self.familias = list(familias)

    def __call__(self, base):
        return pd.concat([base, *[f.aplicar(base) for f in self.familias]], axis=1)


def ajustar_familias(treino, familias):
    """Ajusta as famílias no treino do modelo final.

    Devolve (treino com as colunas novas, AplicarFamilias para escorar outras bases).
    No treino, a taxa por prefixo de CEP continua vindo das outras dobras.
    """
    novas = [f.ajustar_aplicar(treino) for f in familias]
    return pd.concat([treino, *novas], axis=1), AplicarFamilias(familias)
