"""Ajuste de hiperparâmetros por validação cruzada DENTRO do treino.

[IA] Módulo gerado com auxílio de IA (Claude) em 12/10/2026. Registrar em docs/uso_ia.md.

Regras (spec do bloco B, e o ponto fraco do grupo de 2025, que ajustou no teste):
- O teste e a OOT interna não entram aqui. Só o treino (202301-202308, sem o teste).
- 4 dobras agrupadas por `id` (o mesmo cliente nunca fica dos dois lados) e
  estratificadas pelo target, a mesma lógica de `validacao.dividir_treino_teste`.
- Tudo que aprende dos dados (categorias, binning do WoE) é ajustado no treino
  de cada dobra e aplicado na validação da dobra.
- Boosting: parada antecipada pela AUC da validação da dobra. O número de
  árvores final é a média das melhores iterações das dobras. Isso deixa a AUC da
  validação cruzada um pouco otimista, igual para todos os candidatos; a
  comparação justa entre os candidatos finais é feita depois, no teste.
- Busca com Optuna (amostrador TPE com semente fixa). Cada tentativa vira uma linha
  na tabela devolvida; o notebook grava a tabela em `reports/tables/`.
"""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import RANDOM_SEED
from .experimentos import (MODELOS, PARAMS_REFERENCIA, categorias_do_treino, params_do_estimador,
                           pesos_de_classe, preparar_x)
from .validacao import _metricas

N_MAX_ARVORES = 3000


def dobras_por_id(treino, n_dobras=4, semente=RANDOM_SEED, col_target="adimplencia",
                  col_grupo="id"):
    """Lista de (posições de treino, posições de validação), agrupada por id."""
    sgkf = StratifiedGroupKFold(n_splits=n_dobras, shuffle=True, random_state=semente)
    dobras = list(sgkf.split(treino, treino[col_target], groups=treino[col_grupo]))
    grupos = treino[col_grupo].to_numpy()
    for a, b in dobras:
        assert not set(grupos[a]) & set(grupos[b]), "id nos dois lados da dobra"
    return dobras


def _ajustar_na_dobra(modelo, params, x_tr, y_tr, x_va, y_va, semente, parada_antecipada):
    """Treina um estimador numa dobra. Devolve (estimador, melhor número de árvores)."""
    estimador = MODELOS[modelo](params_do_estimador(params), semente)
    pesos = pesos_de_classe(y_tr) if params.get("balancear") else None
    if modelo == "lgbm" and parada_antecipada:
        import lightgbm as lgb
        estimador.set_params(n_estimators=N_MAX_ARVORES)
        estimador.fit(x_tr, y_tr, sample_weight=pesos, eval_X=x_va, eval_y=y_va,
                      eval_metric="auc",
                      callbacks=[lgb.early_stopping(parada_antecipada, first_metric_only=True,
                                                    verbose=False)])
        return estimador, int(estimador.best_iteration_)
    if modelo == "xgb" and parada_antecipada:
        estimador.set_params(n_estimators=N_MAX_ARVORES, early_stopping_rounds=parada_antecipada)
        estimador.fit(x_tr, y_tr, sample_weight=pesos, eval_set=[(x_va, y_va)], verbose=False)
        return estimador, int(estimador.best_iteration) + 1
    estimador.fit(x_tr, y_tr, sample_weight=pesos)
    return estimador, None


def avaliar_cv(modelo, params, treino, variaveis, dobras, semente=RANDOM_SEED,
               parada_antecipada=50, col_target="adimplencia"):
    """AUC e KS médios (e desvios) de um modelo nas dobras do treino."""
    params = {**PARAMS_REFERENCIA.get(modelo, {}), **(params or {})}
    variaveis = list(variaveis)
    aucs, kss, arvores = [], [], []
    inicio = time.perf_counter()
    for posicoes_tr, posicoes_va in dobras:
        tr, va = treino.iloc[posicoes_tr], treino.iloc[posicoes_va]
        categoricas = categorias_do_treino(tr, variaveis)
        x_tr, x_va = preparar_x(tr, variaveis, categoricas), preparar_x(va, variaveis, categoricas)
        y_tr, y_va = tr[col_target].astype(int).to_numpy(), va[col_target].astype(int).to_numpy()
        estimador, melhor = _ajustar_na_dobra(modelo, params, x_tr, y_tr, x_va, y_va, semente,
                                              parada_antecipada)
        m = _metricas(y_va, estimador.predict_proba(x_va)[:, 1])
        aucs.append(m["auc"])
        kss.append(m["ks"])
        if melhor is not None:
            arvores.append(melhor)
    return {"auc_cv": float(np.mean(aucs)), "auc_cv_dp": float(np.std(aucs)),
            "ks_cv": float(np.mean(kss)), "ks_cv_dp": float(np.std(kss)),
            "n_estimators": int(round(np.mean(arvores))) if arvores else None,
            "tempo_s": round(time.perf_counter() - inicio, 1)}


# --------------------------------------------------------------------------- #
# Boosting com Optuna
# --------------------------------------------------------------------------- #
def _espaco_lgbm(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 2000, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "balancear": trial.suggest_categorical("balancear", [False, True]),
    }


def _espaco_xgb(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "balancear": trial.suggest_categorical("balancear", [False, True]),
    }


ESPACOS = {"lgbm": _espaco_lgbm, "xgb": _espaco_xgb}


def ajustar_boosting(modelo, treino, variaveis, n_tentativas=40, n_dobras=4, semente=RANDOM_SEED,
                     parada_antecipada=50, timeout=None, espaco=None, dobras=None, mostrar=True):
    """Busca de hiperparâmetros com Optuna, pela AUC média nas dobras do treino.

    Devolve (melhores parâmetros, tabela das tentativas). Os melhores parâmetros
    já trazem o n_estimators da parada antecipada, prontos para `treinar`.
    """
    import optuna

    espaco = espaco or ESPACOS[modelo]
    dobras = dobras if dobras is not None else dobras_por_id(treino, n_dobras, semente)
    linhas = []

    def objetivo(trial):
        params = espaco(trial)
        r = avaliar_cv(modelo, params, treino, variaveis, dobras, semente, parada_antecipada)
        linhas.append({"tentativa": trial.number, **params, **r})
        if mostrar:
            print(f"{modelo} {trial.number + 1}/{n_tentativas}: AUC {r['auc_cv']:.4f} "
                  f"({r['n_estimators']} árvores, {r['tempo_s']:.0f}s)")
        return r["auc_cv"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudo = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=semente))
    estudo.optimize(objetivo, n_trials=n_tentativas, timeout=timeout)

    tabela = pd.DataFrame(linhas).sort_values("auc_cv", ascending=False).reset_index(drop=True)
    melhor = tabela.iloc[0]
    params = {k: _nativo(melhor[k]) for k in espaco_chaves(tabela)}
    params["n_estimators"] = int(melhor["n_estimators"])
    return params, tabela


def espaco_chaves(tabela):
    """Colunas da tabela de tentativas que são parâmetros (e não resultados)."""
    resultados = {"tentativa", "auc_cv", "auc_cv_dp", "ks_cv", "ks_cv_dp", "n_estimators", "tempo_s"}
    return [c for c in tabela.columns if c not in resultados]


def _nativo(v):
    """Tipos do numpy -> tipos do Python (para o JSON do registro)."""
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def comparar_balanceamento(modelo, params, treino, variaveis, n_dobras=4, semente=RANDOM_SEED,
                           parada_antecipada=50, dobras=None):
    """Os mesmos parâmetros com e sem peso de classe, nas dobras do treino."""
    dobras = dobras if dobras is not None else dobras_por_id(treino, n_dobras, semente)
    linhas = []
    for balancear in (False, True):
        p = {**params, "balancear": balancear}
        if parada_antecipada:
            p.pop("n_estimators", None)  # a parada antecipada escolhe de novo
        linhas.append({"balancear": balancear,
                       **avaliar_cv(modelo, p, treino, variaveis, dobras, semente,
                                    parada_antecipada)})
    return pd.DataFrame(linhas).set_index("balancear")


# --------------------------------------------------------------------------- #
# Logística com WoE: grade, com o binning ajustado uma vez por dobra
# --------------------------------------------------------------------------- #
def ajustar_logistica_woe(treino, variaveis, valores_C=(0.001, 0.01, 0.1, 1.0, 10.0),
                          l1_ratios=(0.0, 1.0), limiares=(0.7, 0.8), iv_minimo=0.02,
                          n_dobras=4, semente=RANDOM_SEED, n_jobs=-1, dobras=None,
                          col_target="adimplencia", mostrar=True):
    """Grade de C, L1/L2 e limiar de correlação, pela AUC média nas dobras do treino.

    Em cada dobra, o binning e o IV são ajustados uma vez (no treino da dobra);
    cada limiar só refaz o filtro de correlação. Devolve (melhores parâmetros, tabela).
    """
    from .modelos import LogisticaWoE, TransformadorWoE

    dobras = dobras if dobras is not None else dobras_por_id(treino, n_dobras, semente)
    variaveis = list(variaveis)
    resultados = []
    for i, (posicoes_tr, posicoes_va) in enumerate(dobras, 1):
        tr, va = treino.iloc[posicoes_tr], treino.iloc[posicoes_va]
        categoricas = categorias_do_treino(tr, variaveis)
        x_tr, x_va = preparar_x(tr, variaveis, categoricas), preparar_x(va, variaveis, categoricas)
        y_tr, y_va = tr[col_target].astype(int).to_numpy(), va[col_target].astype(int).to_numpy()
        inicio = time.perf_counter()
        base = TransformadorWoE(iv_minimo, None, n_jobs).fit(x_tr, y_tr)
        if mostrar:
            print(f"dobra {i}: binning em {time.perf_counter() - inicio:.0f}s, "
                  f"{len(base.variaveis_)} variáveis com IV >= {iv_minimo}")
        for limiar in limiares:
            transformador = base.com_limiar(limiar)
            for l1 in l1_ratios:
                for C in valores_C:
                    m = LogisticaWoE(C=C, l1_ratio=l1, iv_minimo=iv_minimo,
                                     limiar_correlacao=limiar, random_state=semente)
                    m.fit(x_tr, y_tr, transformador=transformador)
                    met = _metricas(y_va, m.predict_proba(x_va)[:, 1])
                    resultados.append({"limiar_correlacao": limiar, "l1_ratio": l1, "C": C,
                                       "dobra": i, "auc": met["auc"], "ks": met["ks"],
                                       "n_variaveis": int((m.coeficientes != 0).sum()),
                                       "convergiu": m.convergiu_})
    r = pd.DataFrame(resultados)
    tabela = (r.groupby(["limiar_correlacao", "l1_ratio", "C"])
              .agg(auc_cv=("auc", "mean"), auc_cv_dp=("auc", "std"), ks_cv=("ks", "mean"),
                   n_variaveis=("n_variaveis", "mean"), convergiu=("convergiu", "all"))
              .sort_values("auc_cv", ascending=False).reset_index())
    melhor = tabela[tabela["convergiu"]].iloc[0]
    params = {"C": float(melhor["C"]), "l1_ratio": float(melhor["l1_ratio"]),
              "iv_minimo": iv_minimo, "limiar_correlacao": float(melhor["limiar_correlacao"])}
    return params, tabela
