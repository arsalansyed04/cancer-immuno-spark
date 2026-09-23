#!/usr/bin/env bash
# Download the IEDB public MySQL dump and import it into a local MySQL/MariaDB
# instance so Spark can read it over JDBC.
set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR"

DUMP_URL="https://www.iedb.org/doc/iedb_public.sql.gz"
DUMP_GZ="$DATA_DIR/iedb_public.sql.gz"

DB_NAME="${IEDB_DB_NAME:-iedb_public}"
DB_USER="${IEDB_DB_USER:-spark}"
DB_PASS="${IEDB_DB_PASS:-sparkpw}"

echo "Downloading IEDB dump (this file is large, tens of GB uncompressed) ..."
wget -q --show-progress -O "$DUMP_GZ" "$DUMP_URL"

echo "Ensuring database and user exist ..."
sudo mysql -e "CREATE DATABASE IF NOT EXISTS ${DB_NAME};"
sudo mysql -e "CREATE USER IF NOT EXISTS '${DB_USER}'@'%' IDENTIFIED BY '${DB_PASS}';"
sudo mysql -e "GRANT ALL PRIVILEGES ON ${DB_NAME}.* TO '${DB_USER}'@'%'; FLUSH PRIVILEGES;"

echo "Importing dump into MySQL database '${DB_NAME}' (this can take a while) ..."
gunzip -c "$DUMP_GZ" | mysql -u root "${DB_NAME}"

echo "Import complete. Verifying tables:"
mysql -u "${DB_USER}" -p"${DB_PASS}" "${DB_NAME}" -e "SHOW TABLES;"

cat <<EOF

If table names differ from src/config.py's IEDB_TABLES mapping, update that
file. To inspect a table's columns:
  mysql -u ${DB_USER} -p${DB_PASS} ${DB_NAME} -e "DESCRIBE reference;"
EOF
