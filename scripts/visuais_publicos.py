"""Gera as imagens públicas do README em `docs/imagens/`.

[IA] Script gerado com auxílio de IA (Claude) em 19/10/2026. Registrar em docs/uso_ia.md.

Nada aqui lê a base do datathon, que é confidencial:
- diagramas do método: só as datas das safras e as regras do projeto;
- dados macro: as séries públicas do BCB e do IBGE versionadas em data/external/;
- gráficos do pipeline: as funções do pacote `equifax` rodadas numa base SINTÉTICA,
  gerada aqui com variáveis inventadas (`sint_01`, ...). Cada uma dessas imagens leva
  a marca "dados sintéticos": elas mostram o que o pipeline produz, não os resultados.

Uso (na pasta modelo_datathon, com o ambiente ativo):
    python scripts/visuais_publicos.py
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

from equifax.avaliacao import (grafico_calibracao, grafico_distribuicao_score,  # noqa: E402
                               grafico_ks_por_safra, grafico_roc, simular_politica)
from equifax.config import RANDOM_SEED, ROOT  # noqa: E402
from equifax.experimentos import escorar, treinar  # noqa: E402
from equifax.explicar import (contribuicoes, escolher_perfis, grafico_beeswarm,  # noqa: E402
                              grafico_dependencia, grafico_waterfall)
from equifax.macro import CATALOGO, montar_tabela_macro  # noqa: E402
from equifax.validacao import avaliar, dividir_treino_teste  # noqa: E402

SAIDA = ROOT / "docs" / "imagens"

# Paleta: superfície, tintas e as três primeiras cores categóricas
SUPERFICIE, TINTA, TINTA_2, MUDO = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRADE, EIXO = "#e1e0d9", "#c3c2b7"
AZUL, LARANJA, VERDE_AGUA = "#2a78d6", "#eb6834", "#1baf7a"
AZUL_CLARO, LARANJA_CLARO, VERDE_AGUA_CLARO = "#cde2fb", "#fbd9cb", "#c9eedf"

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "text.color": TINTA, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.edgecolor": EIXO, "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": cycler(color=[AZUL, LARANJA, VERDE_AGUA]),
    "xtick.color": MUDO, "ytick.color": MUDO, "xtick.labelcolor": TINTA_2, "ytick.labelcolor": TINTA_2,
    "grid.color": GRADE, "grid.linewidth": 0.8, "lines.linewidth": 2, "lines.markersize": 6,
    "legend.frameon": False,
})

AVISO_SINTETICO = "Dados sintéticos: mostra o que o pipeline produz, não os resultados do datathon."


def salvar(fig, nome, sintetico=False):
    if sintetico:
        fig.text(0.5, 0.5, "DADOS SINTÉTICOS", rotation=25, fontsize=42, weight="bold",
                 color=MUDO, alpha=0.14, ha="center", va="center")
        fig.text(0.5, -0.01, AVISO_SINTETICO, ha="center", va="top", fontsize=8.5, color=TINTA_2)
    SAIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA / f"{nome}.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("gravado:", nome)


def mes(safra):
    return pd.Timestamp(f"{str(safra)[:4]}-{str(safra)[4:]}-01")


# --------------------------------------------------------------------------- #
# Diagramas do método
# --------------------------------------------------------------------------- #
def diagrama_pipeline():
    """Ordem dos notebooks, com as bases de entrada e a saída."""
    cinza = "#e7e6e1"
    caixas = {  # nome: (x, y, texto, cor)
        "des": (0, 1, "base_des.csv", cinza),
        "n1": (1, 1, "01\nEDA e\ntratamento", AZUL_CLARO),
        "n2": (2, 1, "02\nSeleção de\nvariáveis", AZUL_CLARO),
        "n3": (3, 1, "03\nM1: Modelo\nBase", AZUL_CLARO),
        "n4": (4, 1, "04\nM2: Feature\nengineering", AZUL_CLARO),
        "n6": (5, 1, "06\nM3: Dados\nmacro", AZUL_CLARO),
        "n7": (6, 1, "07\nVencedor e\nescoragem", AZUL_CLARO),
        "sai": (7, 1, "OOT\nescorada", LARANJA_CLARO),
        "n5": (3, 0, "05\nInterpretabilidade e\nensaio da escoragem", AZUL_CLARO),
        "mac": (5, 0, "Séries públicas\ndo BCB e do IBGE", cinza),
        "oot": (6, 0, "base_oot.csv", cinza),
    }
    setas = [("des", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n6"), ("n6", "n7"),
             ("n7", "sai"), ("n3", "n5"), ("mac", "n6"), ("oot", "n7")]
    largura, altura = 0.8, 0.62
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for x, y, texto, cor in caixas.values():
        largura_caixa = 1.5 if "Interpretabilidade" in texto else largura
        ax.add_patch(FancyBboxPatch((x - largura_caixa / 2, y - altura / 2), largura_caixa, altura,
                                    boxstyle="round,pad=0,rounding_size=0.08", facecolor=cor,
                                    edgecolor="none"))
        ax.text(x, y, texto, ha="center", va="center", fontsize=8.8, color=TINTA, linespacing=1.25)
    for origem, destino in setas:
        x0, y0 = caixas[origem][:2]
        x1, y1 = caixas[destino][:2]
        if y0 == y1:      # mesma linha: da borda direita à borda esquerda
            ini, fim = (x0 + largura / 2, y0), (x1 - largura / 2, y1)
        elif y0 > y1:     # desce
            ini, fim = (x0, y0 - altura / 2), (x1, y1 + altura / 2)
        else:             # sobe
            ini, fim = (x0, y0 + altura / 2), (x1, y1 - altura / 2)
        ax.annotate("", xy=fim, xytext=ini,
                    arrowprops=dict(arrowstyle="-|>", color=MUDO, lw=1.3, shrinkA=2, shrinkB=2))
    ax.set_xlim(-0.55, 7.55)
    ax.set_ylim(-0.45, 1.45)
    ax.axis("off")
    ax.set_title("Pipeline: cada notebook lê o que o anterior gravou", loc="left", pad=6)
    salvar(fig, "metodo_pipeline")


def diagrama_validacao():
    """Linha do tempo das safras: que parte de cada protocolo usa cada mês."""
    safras = [202301 + i for i in range(12)]
    linhas = [
        ("Modelos M1, M2 e M3", [(202301, 202308, "treino (75%) + teste (25%), divididos por cliente", AZUL_CLARO),
                                 (202309, 202310, "OOT interna\nconfirma 1 vez", LARANJA_CLARO),
                                 (202311, 202312, "OOT cega\n(sem target)", "#e7e6e1")]),
        ("Validação temporal do macro", [(202301, 202306, "treino", AZUL_CLARO),
                                         (202307, 202308, "avaliação", VERDE_AGUA_CLARO)]),
        ("Modelo entregue", [(202301, 202310, "retreino com as 10 safras, mesmos hiperparâmetros", AZUL_CLARO),
                             (202311, 202312, "escora a\nOOT cega", "#e7e6e1")]),
    ]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for y, (rotulo, blocos) in enumerate(linhas):
        y = len(linhas) - 1 - y
        ax.text(-0.25, y, rotulo, ha="right", va="center", fontsize=10, color=TINTA, weight="semibold")
        for ini, fim, texto, cor in blocos:
            x0 = safras.index(ini)
            largura = safras.index(fim) - x0 + 1
            ax.add_patch(FancyBboxPatch((x0 + 0.05, y - 0.32), largura - 0.1, 0.64,
                                        boxstyle="round,pad=0,rounding_size=0.12",
                                        facecolor=cor, edgecolor="none"))
            ax.text(x0 + largura / 2, y, texto, ha="center", va="center", fontsize=9, color=TINTA)
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.6, len(linhas) - 0.4)
    ax.set_xticks(np.arange(12) + 0.5, [str(s) for s in safras], fontsize=8.5)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    for lado in ("left", "bottom"):
        ax.spines[lado].set_visible(False)
    ax.set_title("Protocolo de validação: o que cada safra faz em cada etapa", pad=12)
    salvar(fig, "metodo_validacao")


def diagrama_disponibilidade(tabela):
    """Para a safra 202303: que mês de cada série o modelo pode usar, e o que ainda não saiu."""
    concessao = pd.Timestamp("2023-03-01")
    series = [("selic_meta", "BR"), ("ipca_mes", "BR"), ("ipca_12m", "BR"), ("inadimplencia_pf", "BR"),
              ("desocupacao_br", "BR"), ("comprometimento_renda", "BR"), ("endividamento_familias", "BR"),
              ("desocupacao_uf", "São Paulo")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for y, (serie, uf) in enumerate(series[::-1]):
        t = tabela[(tabela["serie"] == serie) & (tabela["uf"] == uf)].sort_values("data_referencia")
        usada = t[t["disponivel_em"] <= concessao].iloc[-1]
        proxima = t[t["disponivel_em"] > concessao].iloc[0]
        # A observação que ainda não saiu na data da concessão (cinza)
        ax.plot([proxima["data_referencia"], proxima["disponivel_em"]], [y - 0.18] * 2, color=EIXO, lw=2)
        ax.plot(proxima["data_referencia"], y - 0.18, "o", color=EIXO, ms=5)
        # A observação que o modelo usa (azul): do mês de referência até o dia em que pode ser usada
        ax.plot([usada["data_referencia"], usada["disponivel_em"]], [y + 0.12] * 2, color=AZUL, lw=2.5)
        ax.plot(usada["data_referencia"], y + 0.12, "o", color=AZUL, ms=7)
        ax.plot(usada["disponivel_em"], y + 0.12, "|", color=AZUL, ms=12, mew=2)
        defasagem = CATALOGO[serie].defasagem_meses
        nome = serie + (" (SP)" if uf != "BR" else "")
        regra = "dia seguinte" if defasagem == 0 else f"{defasagem} meses"
        ax.text(pd.Timestamp("2022-05-20"), y, f"{nome}  ·  {regra}", ha="right", va="center", fontsize=9)
    ax.axvline(concessao, color=LARANJA, lw=2)
    ax.annotate("concessão da safra 202303\n(tratada como feita em 01/03/2023)",
                (concessao, 2.5), xytext=(pd.Timestamp("2023-04-20"), 2.5),
                fontsize=9, color=TINTA, va="center",
                arrowprops=dict(arrowstyle="-", color=LARANJA, lw=1))
    ax.set_xlim(pd.Timestamp("2022-05-25"), pd.Timestamp("2023-08-31"))
    ax.set_ylim(-0.7, len(series) - 0.2)
    meses = pd.date_range("2022-07-01", "2023-07-01", freq="2MS")
    nomes = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
    ax.set_xticks(meses, [f"{nomes[m.month - 1]}/{m.strftime('%y')}" for m in meses])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")
    ax.legend(handles=[plt.Line2D([], [], color=AZUL, marker="o", lw=2.5,
                                  label="valor que o modelo usa: do mês de referência até a data em que pode ser usado"),
                       plt.Line2D([], [], color=EIXO, marker="o", lw=2,
                                  label="próxima observação: ainda não divulgada na data da concessão")],
              loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=1, fontsize=8.5)
    ax.set_title("Dados macro sem vazamento: o que o modelo pode usar na safra 202303", pad=10)
    salvar(fig, "metodo_disponibilidade_macro")


# --------------------------------------------------------------------------- #
# Séries macro públicas (BCB e IBGE)
# --------------------------------------------------------------------------- #
def macro_series_nacionais(tabela):
    paineis = [("selic_meta", "Meta Selic (% a.a.)"), ("ipca_12m", "IPCA em 12 meses (%)"),
               ("desocupacao_br", "Desocupação no Brasil (%)"),
               ("inadimplencia_pf", "Inadimplência da carteira PF, BCB (%)")]
    fig, eixos = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True)
    for ax, (serie, titulo) in zip(eixos.flat, paineis):
        t = tabela[(tabela["serie"] == serie) & (tabela["uf"] == "BR")].sort_values("data_referencia")
        ax.axvspan(mes(202301), mes(202311), color=AZUL_CLARO, lw=0)
        ax.axvspan(mes(202311), pd.Timestamp("2024-01-01"), color=LARANJA_CLARO, lw=0)
        ax.plot(t["data_referencia"], t["valor"], color=AZUL, lw=1.8 if serie != "selic_meta" else 1.5,
                drawstyle="steps-post" if serie == "selic_meta" else "default")
        ax.set_title(titulo, fontsize=10.5)
        ax.grid(axis="y")
    fig.legend(handles=[Patch(color=AZUL_CLARO, label="safras de desenvolvimento (jan a out/2023)"),
                        Patch(color=LARANJA_CLARO, label="safras da OOT cega (nov e dez/2023)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle("Séries macro públicas usadas no M3, por mês de referência", x=0.01, ha="left",
                 fontsize=12, weight="semibold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    salvar(fig, "macro_series_nacionais")


def macro_desocupacao_uf(tabela):
    t = tabela[tabela["serie"] == "desocupacao_uf"]
    ref = pd.Timestamp("2023-06-01")  # 2º trimestre de 2023
    uf = t[t["data_referencia"] == ref].set_index("uf")["valor"].sort_values()
    br = tabela[(tabela["serie"] == "desocupacao_br") & (tabela["data_referencia"] == ref)]["valor"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.axvline(br, color=TINTA_2, ls="--", lw=1.2, zorder=1)
    ax.barh(uf.index, uf.values, color=AZUL, height=0.7, zorder=2)
    for i, v in enumerate(uf.values):
        ax.text(v + 0.12, i, f"{v:.1f}".replace(".", ","), va="center", fontsize=8, color=TINTA_2, zorder=3,
                bbox=dict(facecolor=SUPERFICIE, edgecolor="none", pad=0.6))
    ax.text(br + 0.1, len(uf) - 0.4, f"Brasil: {br:.1f}%".replace(".", ","), fontsize=9, color=TINTA_2)
    ax.set_xlabel("Taxa de desocupação (%)")
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.grid(axis="x")
    ax.set_title("Desocupação por UF no 2º trimestre de 2023 (IBGE, PNAD Contínua)", pad=10)
    salvar(fig, "macro_desocupacao_uf")


# --------------------------------------------------------------------------- #
# Base sintética e gráficos do pipeline
# --------------------------------------------------------------------------- #
def base_sintetica(n=40000, semente=7):
    """Base inventada no formato da do desafio: id, safra, variáveis e target.

    As variáveis e os efeitos são inventados aqui e não imitam a base real.
    `sint_03` tem o código especial -1 ("o evento não aconteceu").
    """
    rng = np.random.default_rng(semente)
    x = {f"sint_{i:02d}": rng.normal(size=n).round(3) for i in range(1, 13)}
    x["sint_01"] = (rng.beta(2, 2, n) * 100).round(1)          # tipo percentual, 0 a 100
    x["sint_02"] = rng.poisson(3, n).astype(float)             # contagem
    x["sint_03"] = rng.gamma(2.0, 40.0, n).round(0)            # valor com código especial
    sem_evento = rng.random(n) < 0.3
    x["sint_03"][sem_evento] = -1
    grupo = rng.choice(["A", "B", "C", "D", "E"], n)
    logito = (1.6
              + 0.045 * (x["sint_01"] - 50)
              - 0.45 * (x["sint_02"] - 3)
              + np.where(sem_evento, 0.9, -0.009 * (x["sint_03"] - 80))
              + 1.0 * np.tanh(2 * x["sint_04"])
              - 0.7 * x["sint_05"]
              + 0.6 * (x["sint_06"] > 1)
              + pd.Series(grupo).map({"A": 0.6, "B": 0.2, "C": 0.0, "D": -0.3, "E": -0.7}).to_numpy())
    adimplencia = (rng.random(n) < 1 / (1 + np.exp(-logito))).astype(int)
    return pd.DataFrame({"id": rng.integers(0, int(n * 0.9), n),
                         "safra": rng.choice([202301 + i for i in range(10)], n),
                         **x, "grupo": grupo, "adimplencia": adimplencia})


def graficos_pipeline():
    base = base_sintetica()
    dev = base[base["safra"] <= 202308]
    treino, teste = dividir_treino_teste(dev)
    variaveis = [c for c in base.columns if c.startswith("sint_")] + ["grupo"]
    print(f"base sintética: {len(base)} linhas, inadimplência {1 - base['adimplencia'].mean():.1%}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgbm = treinar("lgbm", None, treino, variaveis, semente=RANDOM_SEED)
        logit = treinar("logit_woe", None, treino, variaveis)
    y = teste["adimplencia"].to_numpy()
    s_lgbm, s_logit = escorar(lgbm, teste), escorar(logit, teste)
    sufixo = " (dados sintéticos)"

    m = {"LightGBM": avaliar(y, s_lgbm, teste["safra"]), "Logística com WoE": avaliar(y, s_logit, teste["safra"])}
    salvar(grafico_ks_por_safra(m, "KS por safra no teste" + sufixo), "sintetico_ks_por_safra", sintetico=True)
    salvar(grafico_roc({"LightGBM": (y, s_lgbm), "Logística com WoE": (y, s_logit)}, "Curva ROC" + sufixo),
           "sintetico_roc", sintetico=True)
    salvar(grafico_calibracao(y, s_lgbm, "Calibração" + sufixo), "sintetico_calibracao", sintetico=True)
    salvar(grafico_distribuicao_score(s_lgbm, teste["safra"], "Distribuição do score por safra" + sufixo),
           "sintetico_distribuicao_score", sintetico=True)

    # Política de crédito: aprovar do maior score para o menor
    taxas = tuple(np.round(np.arange(0.3, 1.0001, 0.05), 2))
    pol = simular_politica(y, s_lgbm, taxas)
    geral = (y == 0).mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(pol.index * 100, pol["inad_aprovados"] * 100, marker="o", color=AZUL)
    ax.axhline(geral, color=MUDO, ls="--", lw=1.2)
    ax.text(31, geral + 0.4, f"aprovando todos: {geral:.1f}%".replace(".", ","), color=TINTA_2, fontsize=9)
    p70 = pol.loc[0.7, "inad_aprovados"] * 100
    ax.annotate(f"aprovando 70%: {p70:.1f}%".replace(".", ","), (70, p70), xytext=(-10, 18),
                textcoords="offset points", ha="right", fontsize=9, color=TINTA,
                arrowprops=dict(arrowstyle="-", color=MUDO, lw=0.8))
    ax.set_xlabel("Clientes aprovados (%), do maior score para o menor")
    ax.set_ylabel("Inadimplência dos aprovados (%)")
    ax.set_ylim(0, geral * 1.25)
    ax.grid(axis="y")
    ax.set_title("Simulação de política de aprovação" + sufixo)
    salvar(fig, "sintetico_politica", sintetico=True)

    # Interpretabilidade: SHAP exato do LightGBM
    contrib, valor_base, x = contribuicoes(lgbm, teste, amostra=3000, semente=RANDOM_SEED)
    salvar(grafico_beeswarm(contrib, valor_base, x, n=10, titulo="Direção e dispersão do efeito" + sufixo),
           "sintetico_shap_beeswarm", sintetico=True)
    fig = grafico_dependencia(contrib, x, "sint_03")
    fig.axes[0].set_title("Dependência de sint_03, com o código especial -1" + sufixo)
    salvar(fig, "sintetico_shap_dependencia", sintetico=True)
    perfis = escolher_perfis(escorar(lgbm, teste.loc[contrib.index]), indice=contrib.index)
    salvar(grafico_waterfall(contrib, valor_base, x, perfis["q05"], n=10,
                             titulo="Por que um cliente de score baixo recebeu esse score" + sufixo),
           "sintetico_shap_waterfall", sintetico=True)


if __name__ == "__main__":
    tabela = montar_tabela_macro(transformacoes=False)
    diagrama_pipeline()
    diagrama_validacao()
    diagrama_disponibilidade(tabela)
    macro_series_nacionais(tabela)
    macro_desocupacao_uf(tabela)
    graficos_pipeline()
