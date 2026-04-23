#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update
sudo apt-get install -y ca-certificates curl git gnupg htop tmux zstd rsync docker.io docker-buildx
sudo systemctl enable --now docker

if [[ "${AIC_BOOTSTRAP_NVIDIA:-1}" != "0" ]]; then
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  distribution="$(
    . /etc/os-release
    echo "${ID}${VERSION_ID}"
  )"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" |
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
fi

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

mkdir -p "$HOME/src" "$HOME/stable-wm/datasets"

if [[ "${AIC_BOOTSTRAP_NVIDIA:-1}" == "0" ]]; then
  echo "Skipping NVIDIA bootstrap."
elif command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi is not available yet. Reboot the VM or check the DLVM driver image." >&2
fi

echo "VM bootstrap complete."
