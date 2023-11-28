#!/bin/bash

PYTHON_VERSION="3.11"

PROJECT_DIR="$(pwd)"

python${PYTHON_VERSION} -m venv venv

source venv/bin/activate

# # Install packages
# pip install -r requirements.txt

SITE_PACKAGES_DIR="venv/lib/python${PYTHON_VERSION}/site-packages"

# Copy 'lymph' and 'lyscripts' folders to the virtual environment's site-packages
cp -rf "lymph" "${SITE_PACKAGES_DIR}/"
cp -rf "lyscripts" "${SITE_PACKAGES_DIR}/"

echo "Virtual environment created and packages installed."
echo " 'lymph' and 'lyscripts' folders copied to the virtual environment's site-packages."
