#!/usr/bin/env bash
# setup.sh – Environment and dataset check for HW5: Image Captioning
set -euo pipefail

RESET="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
RED="\033[91m"; GREEN="\033[92m"; YELLOW="\033[93m"; CYAN="\033[96m"

info()    { echo -e "  ${CYAN}${BOLD}[INFO]${RESET}  $*"; }
ok()      { echo -e "  ${GREEN}${BOLD}[ OK ]${RESET}  $*"; }
warn()    { echo -e "  ${YELLOW}${BOLD}[WARN]${RESET}  $*"; }
error()   { echo -e "  ${RED}${BOLD}[ERR ]${RESET}  $*"; }
hr()      { echo -e "${DIM}$(printf '─%.0s' {1..62})${RESET}"; }

echo ""
echo -e "${CYAN}${BOLD}  HW5 Image Captioning – Setup${RESET}"
hr

# ── 1. Install Python dependencies ──────────────────────────────────────────
echo ""
info "Checking Python packages..."

install_if_missing() {
    local pkg="$1"
    local import_name="${2:-$1}"
    if python -c "import ${import_name}" &>/dev/null; then
        ok "${pkg} already installed."
    else
        warn "${pkg} not found – installing..."
        pip install "${pkg}" --quiet
        ok "${pkg} installed."
    fi
}

install_if_missing torch
install_if_missing torchvision
install_if_missing kaggle
install_if_missing tqdm
install_if_missing Pillow PIL

# ── 2. Check dataset ─────────────────────────────────────────────────────────
echo ""
hr
info "Checking Flickr8k dataset in ../data/ ..."
echo ""

DATA_DIR="$(dirname "$0")/../data"
IMAGES_DIR="${DATA_DIR}/Images"
CAPTIONS_FILE="${DATA_DIR}/captions.txt"

IMAGES_OK=false
CAPTIONS_OK=false

if [ -d "${IMAGES_DIR}" ] && [ "$(ls -A "${IMAGES_DIR}" 2>/dev/null | wc -l)" -gt 0 ]; then
    IMG_COUNT="$(ls "${IMAGES_DIR}" | wc -l | tr -d ' ')"
    ok "Images/ found  (${IMG_COUNT} files)"
    IMAGES_OK=true
else
    warn "Images/ folder is missing or empty."
fi

if [ -f "${CAPTIONS_FILE}" ]; then
    ok "captions.txt found."
    CAPTIONS_OK=true
else
    warn "captions.txt is missing."
fi

# ── 3. Download dataset if missing ───────────────────────────────────────────
if [ "${IMAGES_OK}" = false ] || [ "${CAPTIONS_OK}" = false ]; then
    echo ""
    hr
    warn "Dataset not fully present. Attempting Kaggle CLI download..."
    echo ""

    mkdir -p "${DATA_DIR}"

    # Check kaggle credentials
    KAGGLE_JSON="${HOME}/.kaggle/kaggle.json"
    if [ ! -f "${KAGGLE_JSON}" ]; then
        echo -e "  ${RED}${BOLD}kaggle.json not found at ~/.kaggle/kaggle.json${RESET}"
        echo ""
        echo -e "  To set up Kaggle API credentials:"
        echo "    1. Go to https://www.kaggle.com/settings  →  API  →  'Create New Token'"
        echo "    2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json"
        echo "    3. Run:  chmod 600 ~/.kaggle/kaggle.json"
        echo "    4. Re-run this script."
        echo ""
        echo -e "  Or download manually:"
        echo "    https://www.kaggle.com/datasets/adityajn105/flickr8k"
        echo "    Then unzip into ../data/ so it contains  Images/  and  captions.txt"
        echo ""
        hr
        echo ""
        exit 1
    fi

    chmod 600 "${KAGGLE_JSON}"
    info "Downloading Flickr8k from Kaggle (this may take a few minutes)..."
    kaggle datasets download -d adityajn105/flickr8k -p "${DATA_DIR}" --unzip

    echo ""
    # Verify after download
    if [ -d "${DATA_DIR}/Images" ] && [ -f "${DATA_DIR}/captions.txt" ]; then
        ok "Download complete. Images/ and captions.txt are in ../data/"
    else
        error "Download finished but expected files are missing in ${DATA_DIR}."
        error "Check the Kaggle dataset structure and re-run."
        echo ""
        hr
        echo ""
        exit 1
    fi
    hr
fi

echo ""
hr
echo ""
echo -e "  ${GREEN}${BOLD}Dataset ready.${RESET} Run preprocessing next:"
echo ""
echo "    python preprocessing.py"
echo ""
hr
echo ""
