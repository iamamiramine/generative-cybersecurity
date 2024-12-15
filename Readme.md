# Generative Cybersecurity Application - Setup Guide

This guide explains how to run the Generative Cybersecurity application. The application consists of two main services: a generative module API and a Streamlit frontend interface.

## Prerequisites

Before starting, ensure you have the following installed on your system:
- Docker Engine (version 19.03.0+)
- Docker Compose (version 2.0+)
- NVIDIA GPU with CUDA support (for model inference)
- NVIDIA Container Toolkit (nvidia-docker2)

### 1. Install Docker

#### 1.1 Uninstall all conflicting packages

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
```

#### 1.2 Set up Docker's `apt` repository

```bash
sudo apt-get update
```

```bash
sudo apt-get install ca-certificates curl
```

```bash
sudo install -m 0755 -d /etc/apt/keyrings
```

```bash
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
```

```bash
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

```bash
sudo apt-get update
```

#### 1.3 Install the Docker packages

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

#### 1.4 Verify that the Docker Engine installation is successful by running the `hello-world` image

```bash
sudo docker run hello-world
```

#### 1.5 Run Docker without root privileges

```bash
sudo groupadd docker
```

```bash
sudo usermod -aG docker $USER
```

```bash
newgrp docker
```

#### 1.6 Verify that you can run `docker` commands without `sudo`

```bash
docker run hello-world
```

#### 1.7 Automatically start Docker and containerd on boot

```bash
sudo systemctl enable docker.service
```

```bash
sudo systemctl enable containerd.service
```

##### 1.7.1 To stop this behavior, use `disable` instead

```bash
sudo systemctl disable docker.service
```

```bash
sudo systemctl disable containerd.service
```

### 2. Install NVIDIA Container Toolkit (Nvidia-docker)

#### 2.1 Run the following commands

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```bash
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```bash
sudo apt-get update
```

```bash
sudo apt-get install -y nvidia-container-toolkit
```

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

```bash
sudo systemctl restart docker
```

#### 2.2 Make sure Nvidia-docker is installed

##### 2.2.1 Nvidia Container Toolkit

```bash
dpkg -l | grep nvidia-container-toolkit
```

Expected Output

```bash
ii  nvidia-container-toolkit                   1.16.2-1                                amd64        NVIDIA Container toolkit
ii  nvidia-container-toolkit-base              1.16.2-1                                amd64        NVIDIA Container Toolkit Base
```

##### 2.2.2 Verify the NVIDIA runtime is available to Docker

```bash
sudo docker info | grep Runtimes
```

Expected Output

```bash
Runtimes: io.containerd.runc.v2 nvidia runc
```

## Project Structure

The application consists of two main components:
- `generative-module-api`: The backend service running the AI model
- `streamlit-module-api`: The frontend service providing the user interface

## Setup Instructions

### 1. Create Docker Network

First, create the required Docker network:
```bash
docker network create generative-cybersecurity-network
```

### 2. Prepare Directory Structure

Ensure you have the following directory structure:

generative-cybersecurity

├── models/                    # Contains AI models

├── data/                     # Contains application data 

├── shared/                   # Shared resources

│   ├── assets/              # Images and other assets

│   └── config/              # Configuration files

├── generative-module-api/    # Backend service

└── streamlit-module-api/     # Frontend service


### 3. Data Setup

Create a directory named "data" in the root directory of the project. Place your txt files related to bash scripts and cybersecurity in this directory.

The directory structure should look like:
generative-cybersecurity/data/

Note: Only txt files are supported for now. The data directory must be named exactly as "data" since this path is referenced in the docker-compose.yml file for volume mounting.


### 4. Model Setup

First, create a "models" directory in the root folder of the project if it doesn't already exist. This is where the model will be downloaded.

To download the required model, make a POST request to the `/download_llm` endpoint of the generative module API:

```bash
curl -X POST http://localhost:7575/download_llm -H "Content-Type: application/json" -d '{"model_name": "WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B"}'
```

This will download the model to the following path:

```bash
generative-cybersecurity/models/WhiteRabbitNeo_WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B/
```

or use Swagger UI to download the model:

```bash
http://localhost:7575/docs#/
```

### 5. Configuration

Ensure your `shared/config/api_config.json` is properly configured with the correct endpoints:

```json
{
"generative_module": "http://generative-module-api:80"
}
```

Note: generative-module-api is the name of the service in the docker-compose.yml file.

### 6. Running the Application

To create the docker network, run:
(Note: this step is only required if you are running the application for the first time.)
```bash
docker network create generative-cybersecurity-network
```
Note: the network name must match the network name in the docker-compose.yml file.

To build the application, run:

```bash
docker compose build
```

To run the application, run:

```bash
docker compose up
```

This will:
- Build and start both services
- Mount necessary volumes
- Configure GPU access for the generative module
- Map ports for both services

### 7. Accessing the Application

Once running, you can access:
- Streamlit UI: `http://0.0.0.0:7500`
- Generative API: `http://0.0.0.0:7575/docs` (Swagger UI)

### 8. Service Ports

- Streamlit Frontend: Port 7500
- Generative API: Port 7575

The ports are pre-defined in the docker-compose.yml file.

### 9. Running FastAPI Endpoints

Access the FastAPI endpoints using the following URL:

```bash
http://localhost:7575/docs
```

This will provide a Swagger UI for testing the FastAPI endpoints.

The sequential order of the endpoints is as follows:

#### No Context

1. Load the model
```bash
POST /load_model
```
2. Load the pipeline
```bash
POST /load_pipeline
```
3. Load Chain
```bash
POST /load_chain
```
4. Generate text
```bash
POST /generate
```

#### With Context

1. Load the model
```bash
POST /load_model
```
2. Load the pipeline
```bash
POST /load_pipeline
```
3. Load Documents
```bash
POST /load_docs
```
4. Load Ensemble Retriever
```bash
POST /load_ensemble_retriever_from_docs
```
5. Load Chain
```bash
POST /load_chain
```
6. Generate text
```bash
POST /generate
```


### 10. Streamlit UI

To access the Streamlit UI, run:

```bash
http://localhost:7500
```

This will provide a user interface for interacting with the application.

The application will call the following endpoints in order:

1. Load the model
2. Load the pipeline
3. Load the documents
4. Load the ensemble retriever
5. Load the chain
6. Generate text

At every context change, the application will reload:
1. The ensemble retriever
2. The chain

At every change of parameters, the application will reload:
1. The pipeline
2. The chain

