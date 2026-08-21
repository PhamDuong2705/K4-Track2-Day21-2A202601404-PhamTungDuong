#!/usr/bin/env bash
# Run once on a fresh Ubuntu EC2 instance:
#   bash deploy/bootstrap-ec2.sh <S3_BUCKET> [PUBLIC_GIT_REPOSITORY]

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <S3_BUCKET> [PUBLIC_GIT_REPOSITORY]" >&2
  exit 2
fi

ARTIFACT_BUCKET="$1"
REPOSITORY_URL="${2:-https://github.com/PhamDuong2705/K4-Track2-Day21-2A202601404-PhamTungDuong.git}"
if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
  APP_USER="$SUDO_USER"
elif id ubuntu >/dev/null 2>&1; then
  APP_USER="ubuntu"
else
  APP_USER="$USER"
fi
APP_HOME="$(getent passwd "$APP_USER" | cut -d: -f6)"
APP_DIR="$APP_HOME/income-api"

sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git curl

if [[ ! -d "$APP_DIR/.git" ]]; then
  git clone --branch main "$REPOSITORY_URL" "$APP_DIR"
else
  git -C "$APP_DIR" pull --ff-only origin main
fi

python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"
mkdir -p "$APP_HOME/models"
sudo chown -R "$APP_USER:$APP_USER" "$APP_DIR" "$APP_HOME/models"

sudo tee /etc/systemd/system/income-api.service >/dev/null <<EOF
[Unit]
Description=Adult Income FastAPI inference service
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
Environment="ARTIFACT_BUCKET=$ARTIFACT_BUCKET"
Environment="MODEL_PATH=$APP_HOME/models/model.joblib"
ExecStart=$APP_DIR/.venv/bin/python $APP_DIR/src/serve.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable income-api

echo "EC2 bootstrap complete. The first CI release will upload the model and restart income-api."
echo "Verify later with: curl http://localhost:8080/healthz"
