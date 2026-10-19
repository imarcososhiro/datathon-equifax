"""Testes da coleta macro (src/equifax/macro.py), com respostas de API inventadas.

[IA] Testes gerados com auxílio de IA (Claude) em 11/10/2026. Registrar em docs/uso_ia.md.
Nenhum teste acessa a rede: o download é substituído por uma função falsa.
"""
import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from equifax.macro import (CATALOGO, baixar_catalogo, ler_bruto, ler_resposta_sgs,
                           ler_resposta_sidra, resumo, url_sgs, url_sidra)

CAB_TRIMESTRAL = {"NC": "Nível Territorial (Código)", "NN": "Nível Territorial",
                  "MC": "Unidade de Medida (Código)", "MN": "Unidade de Medida", "V": "Valor",
                  "D1C": "Unidade da Federação (Código)", "D1N": "Unidade da Federação",
                  "D2C": "Variável (Código)", "D2N": "Variável",
                  "D3C": "Trimestre (Código)", "D3N": "Trimestre"}
CAB_MOVEL = {**CAB_TRIMESTRAL, "D1C": "Brasil (Código)", "D1N": "Brasil",
             "D3C": "Trimestre Móvel (Código)", "D3N": "Trimestre Móvel"}


def _linha_sidra(uf_cod, uf, periodo, valor):
    return {"NC": "3", "NN": "x", "MC": "2", "MN": "%", "V": valor, "D1C": uf_cod, "D1N": uf,
            "D2C": "4099", "D2N": "x", "D3C": periodo, "D3N": "x"}


def _resposta_falsa(url):
    """Imita as APIs: devolve o JSON conforme a URL."""
    if "sgs.432/" in url:
        return [{"data": "28/02/2023", "valor": "13.75"}, {"data": "01/03/2023", "valor": "13.75"}]
    if "sgs.433/" in url:
        return [{"data": "01/01/2023", "valor": "0.53"}, {"data": "01/02/2023", "valor": "0.84"}]
    if "sgs.13522/" in url:
        return [{"data": "01/12/2022", "valor": "5.79"}]
    if "sgs." in url:
        return [{"data": "01/01/2023", "valor": "3.9"}]
    if "/t/6381/" in url:
        return [CAB_MOVEL, _linha_sidra("1", "Brasil", "202308", "7.8")]
    if "/t/4099/" in url:
        return [CAB_TRIMESTRAL, _linha_sidra("35", "São Paulo", "202302", "7.8"),
                _linha_sidra("33", "Rio de Janeiro", "202302", "11.3"),
                _linha_sidra("33", "Rio de Janeiro", "201901", "15.0")]  # fora da janela
    raise AssertionError(f"URL inesperada: {url}")


# --- urls --------------------------------------------------------------------

def test_urls():
    assert url_sgs("432", date(2020, 1, 1), date(2024, 12, 31)) == (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados?formato=json"
        "&dataInicial=01%2F01%2F2020&dataFinal=31%2F12%2F2024")
    assert url_sidra(CATALOGO["desocupacao_uf"]) == (
        "https://apisidra.ibge.gov.br/values/t/4099/n3/all/v/4099/p/all")


# --- leitura das respostas -------------------------------------------------------

def test_sgs_mensal_vira_primeiro_dia_do_mes():
    t = ler_resposta_sgs([{"data": "01/01/2023", "valor": "0.53"}], CATALOGO["ipca_mes"])
    assert t.loc[0, "data_referencia"] == date(2023, 1, 1)
    assert t.loc[0, "valor"] == pytest.approx(0.53)
    assert t.loc[0, "territorio_codigo"] == "BR"


def test_sgs_diaria_mantem_o_dia():
    t = ler_resposta_sgs([{"data": "28/02/2023", "valor": "13.75"}], CATALOGO["selic_meta"])
    assert t.loc[0, "data_referencia"] == date(2023, 2, 28)


def test_sgs_resposta_de_erro_da_mensagem_clara():
    with pytest.raises(ValueError, match="Resposta inesperada"):
        ler_resposta_sgs({"erro": "serie inexistente"}, CATALOGO["ipca_mes"])


def test_sidra_trimestral_usa_o_ultimo_mes_do_trimestre():
    dados = [CAB_TRIMESTRAL, _linha_sidra("35", "São Paulo", "202301", "8.5"),
             _linha_sidra("35", "São Paulo", "202304", "...")]
    t = ler_resposta_sidra(dados, CATALOGO["desocupacao_uf"])
    assert list(t["data_referencia"]) == [date(2023, 3, 1), date(2023, 12, 1)]
    assert t.loc[0, "valor"] == pytest.approx(8.5)
    assert pd.isna(t.loc[1, "valor"])  # "..." vira ausente
    assert t.loc[0, "territorio"] == "São Paulo"


def test_sidra_trimestre_movel_usa_o_mes_do_codigo():
    t = ler_resposta_sidra([CAB_MOVEL, _linha_sidra("1", "Brasil", "202308", "7.8")],
                           CATALOGO["desocupacao_br"])
    assert t.loc[0, "data_referencia"] == date(2023, 8, 1)
    assert t.loc[0, "territorio_codigo"] == "BR"


def test_sidra_trimestre_invalido_da_erro():
    with pytest.raises(ValueError, match="Trimestre inválido"):
        ler_resposta_sidra([CAB_TRIMESTRAL, _linha_sidra("35", "SP", "202305", "1")],
                           CATALOGO["desocupacao_uf"])


# --- coleta completa ----------------------------------------------------------------

def test_baixar_catalogo_grava_arquivos_e_manifesto(tmp_path):
    agora = datetime(2026, 10, 11, 20, 0, 0)
    destino, manifesto = baixar_catalogo(tmp_path, baixar=_resposta_falsa, agora=agora)
    assert destino == tmp_path / "2026-10-11"
    assert sorted(p.name for p in destino.glob("*.csv")) == sorted(f"{n}.csv" for n in CATALOGO)
    lido = json.loads((destino / "manifesto.json").read_text(encoding="utf-8"))
    assert lido["baixado_em"] == "2026-10-11T20:00:00"
    uf = lido["series"]["desocupacao_uf"]
    assert uf["n_linhas"] == 2 and uf["n_territorios"] == 2  # 2019 ficou fora da janela
    assert uf["url"].endswith("/t/4099/n3/all/v/4099/p/all")
    assert uf["catalogo"]["defasagem_meses"] == 4
    bruto = ler_bruto(destino)
    assert set(bruto["serie"]) == set(CATALOGO)


def test_uma_serie_com_erro_nao_para_as_outras(tmp_path):
    def falha_no_ipca(url):
        if "sgs.433/" in url:
            raise RuntimeError("timeout")
        return _resposta_falsa(url)
    destino, manifesto = baixar_catalogo(tmp_path, baixar=falha_no_ipca)
    assert "timeout" in manifesto["series"]["ipca_mes"]["erro"]
    assert not (destino / "ipca_mes.csv").exists()
    assert (destino / "ipca_12m.csv").exists()


def test_resumo_confere_valores_conhecidos(tmp_path):
    destino, _ = baixar_catalogo(tmp_path, baixar=_resposta_falsa)
    r = resumo(destino).set_index("serie")
    assert r.loc["selic_meta", "conferencia"].startswith("ok")
    assert r.loc["ipca_12m", "conferencia"].startswith("ok")
    assert r.loc["ipca_mes", "conferencia"].startswith("ok")


def test_resumo_acusa_codigo_errado(tmp_path):
    def selic_errada(url):
        if "sgs.432/" in url:
            return [{"data": "01/03/2023", "valor": "5.0"}]
        return _resposta_falsa(url)
    destino, _ = baixar_catalogo(tmp_path, baixar=selic_errada)
    assert resumo(destino).set_index("serie").loc["selic_meta", "conferencia"].startswith("CONFERIR")


def test_catalogo_coerente():
    for s in CATALOGO.values():
        assert s.fonte in ("sgs", "sidra")
        assert s.geografia in ("BR", "UF")
        assert s.defasagem_meses >= (0 if s.frequencia == "diaria" else 2)
        if s.fonte == "sidra":
            assert s.variavel_sidra and s.nivel_sidra in ("n1", "n3")
            assert (s.nivel_sidra == "n3") == (s.geografia == "UF")


# --- junção sem vazamento (bloco D, item 3) -------------------------------------------
# [IA] Testes gerados com auxílio de IA (Claude) em 13/10/2026.

from equifax.macro import (adicionar_macro, data_da_concessao, disponivel_em, juntar_macro,
                           montar_tabela_macro, JuntarMacro)


def _bruto(tmp_path):
    """Pasta com séries inventadas: IPCA mensal, Selic diária e desocupação por UF e Brasil."""
    pasta = tmp_path / "2026-10-12"
    pasta.mkdir()
    meses = pd.date_range("2022-01-01", "2023-12-01", freq="MS")
    pd.DataFrame({"serie": "ipca_12m", "periodo_codigo": meses.strftime("%d/%m/%Y"),
                  "data_referencia": meses.date, "territorio_codigo": "BR", "territorio": "Brasil",
                  "valor": range(len(meses))}).to_csv(pasta / "ipca_12m.csv", index=False)
    dias = pd.to_datetime(["2023-02-27", "2023-02-28", "2023-03-01", "2023-03-02"])
    pd.DataFrame({"serie": "selic_meta", "periodo_codigo": dias.strftime("%d/%m/%Y"),
                  "data_referencia": dias.date, "territorio_codigo": "BR", "territorio": "Brasil",
                  "valor": [13.0, 13.25, 99.0, 99.0]}).to_csv(pasta / "selic_meta.csv", index=False)
    trimestres = pd.date_range("2021-03-01", "2023-12-01", freq="3MS")
    linhas = []
    for uf, base in [("São Paulo", 10.0), ("Bahia", 20.0)]:
        for i, ref in enumerate(trimestres):
            linhas.append({"serie": "desocupacao_uf", "periodo_codigo": "x", "data_referencia": ref.date(),
                           "territorio_codigo": "35", "territorio": uf, "valor": base + i})
    pd.DataFrame(linhas).to_csv(pasta / "desocupacao_uf.csv", index=False)
    pd.DataFrame({"serie": "desocupacao_br", "periodo_codigo": "x", "data_referencia": meses.date,
                  "territorio_codigo": "BR", "territorio": "Brasil",
                  "valor": [50.0 + i for i in range(len(meses))]}).to_csv(pasta / "desocupacao_br.csv",
                                                                         index=False)
    return pasta


def test_disponivel_em_segue_a_defasagem():
    mensal = disponivel_em(["2023-01-01"], CATALOGO["ipca_12m"])          # defasagem 2
    diaria = disponivel_em(["2023-02-28"], CATALOGO["selic_meta"])        # dia seguinte
    trimestral = disponivel_em(["2022-09-01"], CATALOGO["desocupacao_uf"])  # defasagem 4
    assert pd.Timestamp(mensal[0]) == pd.Timestamp("2023-03-01")
    assert pd.Timestamp(diaria[0]) == pd.Timestamp("2023-03-01")
    assert pd.Timestamp(trimestral[0]) == pd.Timestamp("2023-01-01")


def test_concessao_no_dia_1_da_safra():
    assert list(data_da_concessao([202303, 202312])) == [pd.Timestamp("2023-03-01"),
                                                         pd.Timestamp("2023-12-01")]


def test_valor_divulgado_depois_do_dia_1_nunca_entra(tmp_path):
    tabela = montar_tabela_macro(_bruto(tmp_path), transformacoes=False)
    base = pd.DataFrame({"safra": [202303], "estado": ["São Paulo"]})
    j = juntar_macro(base, tabela)
    # IPCA: em 01/03/2023 só o de janeiro (índice 12) estava disponível; o de fevereiro, não
    assert j.loc[0, "macro_ipca_12m"] == 12
    assert j.loc[0, "macro_ipca_12m_idade_meses"] == 2
    # Selic: o valor de 01/03 só vale a partir de 02/03; entra o de 28/02
    assert j.loc[0, "macro_selic_meta"] == 13.25
    # Desocupação por UF: o 3º trimestre de 2022 (vale desde 01/01/2023); o 4º só em 01/04
    assert j.loc[0, "macro_desocupacao_uf"] == 10.0 + 6
    assert j.loc[0, "macro_desocupacao_uf_idade_meses"] == 6


def test_juncao_nao_duplica_nem_perde_linhas_e_mantem_a_ordem(tmp_path):
    tabela = montar_tabela_macro(_bruto(tmp_path))
    base = pd.DataFrame({"safra": [202305, 202301, 202303, 202305],
                         "estado": ["Bahia", "São Paulo", "Sem informação", "Bahia"]},
                        index=[40, 10, 30, 20])
    j = juntar_macro(base, tabela)
    assert list(j.index) == [40, 10, 30, 20]
    assert j.loc[40].equals(j.loc[20])  # mesma safra e mesmo estado, mesmos valores
    assert j.loc[10, "macro_desocupacao_uf"] != j.loc[40, "macro_desocupacao_uf"]


def test_sem_informacao_recebe_o_valor_nacional(tmp_path):
    tabela = montar_tabela_macro(_bruto(tmp_path))
    base = pd.DataFrame({"safra": [202303], "estado": ["Sem informação"]})
    j = juntar_macro(base, tabela)
    assert j.loc[0, "macro_desocupacao_uf"] == j.loc[0, "macro_desocupacao_br"]
    assert j.loc[0, "macro_desocupacao_uf_menos_br"] == 0.0


def test_estado_desconhecido_da_erro(tmp_path):
    tabela = montar_tabela_macro(_bruto(tmp_path))
    with pytest.raises(ValueError, match="Atlântida"):
        juntar_macro(pd.DataFrame({"safra": [202303], "estado": ["Atlântida"]}), tabela)


def test_transformacoes(tmp_path):
    tabela = montar_tabela_macro(_bruto(tmp_path))
    var = tabela[tabela["serie"] == "ipca_12m_var_3m"].set_index("data_referencia")["valor"]
    assert (var == 3).all()  # a série inventada sobe 1 por mês
    dif = tabela[(tabela["serie"] == "desocupacao_uf_menos_br") & (tabela["uf"] == "Bahia")]
    linha = dif[dif["data_referencia"] == pd.Timestamp("2022-09-01")].iloc[0]
    assert linha["valor"] == (20.0 + 6) - (50.0 + 8)  # Bahia no 3º tri de 2022 menos Brasil em 09/2022
    # vale só quando as duas séries valem (a da UF tem a defasagem maior)
    assert linha["disponivel_em"] == pd.Timestamp("2023-01-01")


def test_adicionar_macro_altera_a_base(tmp_path):
    tabela = montar_tabela_macro(_bruto(tmp_path), transformacoes=False)
    base = pd.DataFrame({"safra": [202303, 202304], "estado": ["Bahia", "São Paulo"]})
    adicionar_macro(base, tabela, idade=False)
    assert "macro_ipca_12m" in base.columns and len(base) == 2
    assert not any(c.endswith("_idade_meses") for c in base.columns)


def test_serie_sem_dado_na_data_avisa(tmp_path):
    tabela = montar_tabela_macro(_bruto(tmp_path), transformacoes=False)
    base = pd.DataFrame({"safra": [202201], "estado": ["Bahia"]})  # antes de qualquer divulgação
    with pytest.warns(UserWarning, match="ausentes"):
        j = juntar_macro(base, tabela, series=["ipca_12m"])
    assert j["macro_ipca_12m"].isna().all()


def test_estado_como_categoria(tmp_path):
    tabela = montar_tabela_macro(_bruto(tmp_path))
    base = pd.DataFrame({"safra": [202303, 202303],
                         "estado": pd.Categorical(["Bahia", np.nan])})
    j = juntar_macro(base, tabela)
    assert j.loc[1, "macro_desocupacao_uf"] == j.loc[1, "macro_desocupacao_br"]


def test_juntar_macro_do_pacote_sobrevive_ao_joblib(tmp_path):
    import joblib
    tabela = montar_tabela_macro(_bruto(tmp_path))
    base = pd.DataFrame({"safra": [202303, 202305], "estado": ["Bahia", "Sem informação"]})
    joblib.dump(JuntarMacro(tabela, ["desocupacao_uf", "desocupacao_uf_menos_br"]), tmp_path / "j.joblib")
    juntar = joblib.load(tmp_path / "j.joblib")
    saida = juntar(base)
    esperado = juntar_macro(base, tabela, series=["desocupacao_uf", "desocupacao_uf_menos_br"])
    pd.testing.assert_frame_equal(saida[esperado.columns], esperado)
    assert list(saida.columns[:2]) == ["safra", "estado"] and len(saida) == 2
