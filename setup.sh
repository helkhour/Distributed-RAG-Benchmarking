#!/bin/bash

# Update package lists
sudo apt-get update
sudo apt-get install -y gnupg curl ca-certificates python3-venv python3-dev build-essential

# Install NVIDIA Drivers
# Required for PyTorch to use T4 GPU on G4dn.xlarge
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo apt-get install -y nvidia-driver-535 nvidia-utils-535

# MongoDB Atlas CLI
curl -fsSL https://pgp.mongodb.com/server-7.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt-get update
sudo apt-get install -y mongodb-atlas-cli

# Docker (for MongoDB Atlas local deployment)
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove -y $pkg; done
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker ubuntu


pip install bitsandbytes

# Python
cd /home/ubuntu/rag_project
# Ensure python3 is valid
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi
python3 --version
# Create virtual environment
rm -rf venv  # Clean any broken venv
python3 -m venv venv
source venv/bin/activate
# Verify pip exists
if ! command -v pip &> /dev/null; then
    echo "Error: pip not found in virtual environment"
    deactivate
    exit 1
fi
# Install Hugging Face CLI and requirements
pip install huggingface_hub[cli]
pip install -r requirements.txt
deactivate

# Set up Hugging Face token
mkdir -p ~/.huggingface
echo "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" > ~/.huggingface/token
chmod 600 ~/.huggingface/token

# Clean up
sudo apt-get autoremove -y
sudo apt-get clean