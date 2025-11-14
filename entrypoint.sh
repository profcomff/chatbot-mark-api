#!/bin/bash
set -e

# Загрузка сертификата при старте
mkdir -p /app/llm
curl -s -o /app/llm/russian_trusted_root_ca.crt "${CA_BUNDLE_URL}"

# Опционально: установка в системное хранилище
cp /app/llm/russian_trusted_root_ca.crt /usr/local/share/ca-certificates/
update-ca-certificates

exec "$@"
