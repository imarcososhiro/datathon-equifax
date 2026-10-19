"""Base inventada compartilhada pelos testes do motor de experimentos.

[IA] Gerado com auxílio de IA (Claude) em 11/10/2026. Registrar em docs/uso_ia.md.
"""
import numpy as np
import pandas as pd
import pytest


def _base_inventada(n=3000, semente=0):
    """10 safras, ids repetidos, VPs com códigos especiais, categóricas e poucos maus."""
    rng = np.random.default_rng(semente)
    vps = {f"VP{i:02d}": rng.normal(size=n).round(2) for i in range(1, 7)}
    vps["VP01"][rng.random(n) < 0.1] = -1
    vps["VP02"][rng.random(n) < 0.05] = 9999
    vps["VP06"] = rng.integers(0, 500, n).astype(float)  # inteiros: cabem em float32
    logito = 2.2 + 0.8 * vps["VP03"] - 0.6 * vps["VP04"]
    adimplencia = (rng.random(n) < 1 / (1 + np.exp(-logito))).astype(int)
    faixa = rng.choice(["3 - 1000", "1001 - 2000", "2001 - 3000"], n).astype(object)
    faixa[rng.random(n) < 0.02] = np.nan
    return pd.DataFrame({
        "id": rng.integers(0, 2500, n),
        "safra": rng.choice(range(202301, 202311), n),
        "adimplencia": adimplencia,
        "faixa_renda": faixa,
        "estado": rng.choice(["São Paulo", "Bahia", "Paraná", "Sem informação"], n),
        "faixa_idade": rng.choice(["18 - 25", "26 - 35", "36 - 45", "Sem informação"], n),
        "cep8": rng.choice(["01310100", "20040002", "Sem informação"], n),
        **vps,
    })


@pytest.fixture
def base():
    return _base_inventada()
