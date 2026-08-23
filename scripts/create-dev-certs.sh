#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cert_dir="${repo_root}/certs"
mkdir -p "${cert_dir}"

openssl req -x509 -newkey rsa:2048 -sha256 -days 825 -nodes \
  -keyout "${cert_dir}/localhost-key.pem" \
  -out "${cert_dir}/localhost.pem" \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1" \
  -addext "keyUsage=digitalSignature,keyEncipherment" \
  -addext "extendedKeyUsage=serverAuth"

chmod 600 "${cert_dir}/localhost-key.pem"
printf 'Created %s and %s\n' \
  "${cert_dir}/localhost.pem" "${cert_dir}/localhost-key.pem"
