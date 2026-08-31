#!/usr/bin/env bash
set -euo pipefail

if [ -z "${DAGSHUB_USERNAME:-}" ] || [ -z "${DAGSHUB_TOKEN:-}" ]; then
  echo "ERROR: DAGSHUB_USERNAME and DAGSHUB_TOKEN must be set (see .env.example)." >&2
  exit 1
fi

if [ ! -d ".dvc" ]; then
  dvc init
else
  echo ".dvc already initialized, skipping dvc init."
fi

DAGSHUB_REPO_URL="https://dagshub.com/${DAGSHUB_USERNAME}/research_project.dvc"

dvc remote add -f dagshub "$DAGSHUB_REPO_URL"
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user "$DAGSHUB_USERNAME"
dvc remote modify dagshub --local password "$DAGSHUB_TOKEN"
dvc remote default dagshub

echo "DVC + DagsHub remote configured. Run 'dvc push' after 'dvc add'/'dvc repro' to sync data."