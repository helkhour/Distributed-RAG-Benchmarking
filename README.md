# RAG Project Setup on AWS VM (Ubuntu)

This README provides a step-by-step guide to set up an AWS Ubuntu VM for the `rag_project`. It includes syncing project data, installing dependencies (MongoDB Atlas CLI, Docker), configuring Atlas deployments, and installing Python packages from `requirements.txt`.

---

## Prerequisites

- An AWS EC2 instance running Ubuntu (e.g., Ubuntu 22.04 Jammy)
- SSH access to the VM (e.g., `ssh ubuntu@<VM_IP>`)
- Local `rag_project` directory at `/projects/rag_project/`
- A `requirements.txt` file in the `rag_project` directory

---

## Setup Instructions

### 1. Sync Project Data

```bash
rsync -avz /projects/rag_project/ ubuntu@<VM_IP>:/home/ubuntu/rag_project
```

> Replace `<VM_IP>` with your VM’s public IP.  
> Ensure the local path matches your setup.

---

### 2. Install MongoDB Atlas CLI

#### Update Package List

```bash
sudo apt-get update
sudo apt-get install -y gnupg curl
```

#### Add MongoDB GPG Key

```bash
curl -fsSL https://pgp.mongodb.com/server-7.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
```

#### Add MongoDB Repository

```bash
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | \
   sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
```

#### Install Atlas CLI

```bash
sudo apt-get update
sudo apt-get install -y mongodb-atlas-cli
```

#### Verify Installation

```bash
atlas --version
```

> If this fails, ensure the package installed correctly or check the [MongoDB Atlas CLI Docs](https://www.mongodb.com/docs/atlas/cli/current/install-atlas-cli/)

---

### 3. Install Docker

#### Remove Old Docker Packages

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove -y $pkg; done
```

#### Add Docker GPG Key

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

#### Add Docker Repository

```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"${UBUNTU_CODENAME:-$VERSION_CODENAME}\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

#### Install Docker

```bash
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

#### Test Docker

```bash
sudo docker run hello-world
```

> This should print a "Hello from Docker!" message.

#### (Optional) Run Docker Without sudo

```bash
sudo usermod -aG docker ubuntu
newgrp docker
```

---

### 4. Set Up MongoDB Atlas Local Deployment

#### Create a Local Deployment

```bash
atlas deployments setup --type local
```

> Accept the defaults when prompted.

#### Start the Deployment

```bash
atlas deployments list
atlas deployments start <deployment-name>
```

#### Connect to the Deployment

```bash
atlas deployments connect <deployment-name>
```

---

### 5. Install Python Dependencies

#### Install Virtual Environment Tools

```bash
sudo apt-get install -y python3-venv
```

#### Create and Activate a Virtual Environment

```bash
cd /home/ubuntu/rag_project
python3 -m venv venv
source venv/bin/activate
```

#### Install Requirements

```bash
pip install -r requirements.txt
```

> If you encounter space issues (e.g., `No space left on device`), see [Troubleshooting](#troubleshooting)

#### Deactivate Environment (When Done)

```bash
deactivate
```

---

## Running the Project

```bash
atlas deployments start <deployment-name>
source /home/ubuntu/rag_project/venv/bin/activate
python main.py
```

---

## Troubleshooting

### Disk Space Issues

```bash
df -h
sudo apt clean
sudo apt autoremove --purge
sudo rm -rf /tmp/*
pip cache purge
```

Resize EBS volume from AWS Console, then:

```bash
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
```

---

### Docker Permission Denied

```bash
sudo usermod -aG docker ubuntu
newgrp docker
```

---

### Atlas CLI Not Found

```bash
sudo apt-get install -y mongodb-atlas-cli
```

---

## Automating Setup : 

### 1. Sync your data on the VM 
```bash
rsync -avz /projects/rag_project/ ubuntu@<VM_IP>:/home/ubuntu/rag_project
```
### 1. Make the script setup.sh xecutable

```bash
chmod +x setup.sh
```

### 3. Run the script on the VM

```bash
ssh ubuntu@<VM_IP> "cd /home/ubuntu/rag_project && ./setup.sh"
```
## 4. Setup Atlas deployment on VM

```bash
atlas deployments setup --type local
atlas deployments connect <deloyment_name>
```
---

## Notes

- Replace `<VM_IP>` with your VM’s public IP
- Ensure your VM has **at least 20 GB** of disk space
- This setup assumes Ubuntu 22.04 (Jammy). Adjust if you're using a different version: