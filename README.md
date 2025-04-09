# RAG Project Setup on AWS VM (Ubuntu)

This README provides a step-by-step guide to set up an AWS Ubuntu VM for the `rag_project`. It includes syncing project data, installing dependencies (MongoDB Atlas CLI, Docker), configuring Atlas deployments, and installing Python packages from `requirements.txt`.

---

## Prerequisites

- An AWS EC2 instance running Ubuntu (e.g., Ubuntu 22.04 Jammy) 20 GB root size
- SSH access to the VM (e.g., `ssh ubuntu@<VM_IP>`)
- Local `rag_project` directory at `/projects/rag_project/`

---

## Setup Instructions

### 1. Sync Project Data 

Locally run : 
```bash
rsync -avz /projects/rag_project/ ubuntu@<VM_IP>:/home/ubuntu/rag_project
```
> Pre-requisite : on VM create folder rag_project (mkdir)  
> Replace `<VM_IP>` with your AWS instance VM’s public IP.  

---

### 2. Run the script on the VM in folder rag_project
```bash
./setup.sh
```
---

### 3. Change Docker permissions 

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

#### Activate a Virtual Environment

```bash
cd /home/ubuntu/rag_project
source venv/bin/activate
```
---

## Running the Project

```bash
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
