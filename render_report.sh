#!/usr/bin/env bash
# Executa o notebook ponta-a-ponta sobre o DATASET COMPLETO (1M linhas) e gera o
# relatório HTML com todas as saídas/gráficos embutidos.
#
# Uso (de dentro do container):
#   bash render_report.sh
# Ou a partir do host:
#   docker compose run --rm jupyter bash render_report.sh
set -euo pipefail

NOTEBOOK="${1:-trabalho_cars.ipynb}"

jupyter nbconvert \
    --to html \
    --execute \
    --ExecutePreprocessor.timeout=3600 \
    --output trabalho_cars.html \
    "${NOTEBOOK}"

echo "Relatório gerado: trabalho_cars.html"
