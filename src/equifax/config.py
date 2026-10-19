from pathlib import Path

# Raiz do repositorio (config.py esta em src/equifax/, entao sobe 2 niveis)
ROOT = Path(__file__).resolve().parents[2]

# Dados
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXTERNAL = ROOT / "data" / "external"

# Saidas
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
MODELS = ROOT / "models"
SUBMISSION = ROOT / "submission"

# Arquivos das bases
DES_PATH = DATA_RAW / "base_des.csv"
OOT_PATH = DATA_RAW / "base_oot.csv"

# Base tratada pelo 01_eda e a cópia em parquet (tipos menores, leitura rápida)
DES_TRATADA_PATH = DATA_PROCESSED / "base_des_01.csv"
DES_TRATADA_PARQUET = DATA_INTERIM / "base_des_01.parquet"

# Arquivos gerados pelos notebooks (fora do Git: dependem da base)
CONFIG = ROOT / "config"
CONJUNTOS_PATH = CONFIG / "conjuntos.yaml"
EXPERIMENTOS_PATH = REPORTS / "experimentos.csv"

# Reprodutibilidade (convencao do projeto)
RANDOM_SEED = 42
