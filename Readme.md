<a id="readme-top"></a>
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Generative Cybersecurity</h3>

</div>

# 🚀 **Generative Cybersecurity Application**

An innovative AI-powered learning assistant for solving cybersecurity challenges, leveraging advanced AI models like **White Rabbit** with **Retrieval-Augmented Generation (RAG)** and **Hypothetical Document Embeddings (HyDE)** for accurate, context-aware responses.

---

## 📖 **Table of Contents**
- [About the Project](#about-the-project)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Acknowledgments](#acknowledgments)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 📝 **About the Project**

The **Generative Cybersecurity Application** is designed to enhance learning and problem-solving in cybersecurity using generative AI models. It offers:
- Context-aware assistance for cybersecurity challenges.
- Ethical educational support using authorized materials like **OverTheWire Bandit wargame** series.
- Integration of **AI-driven query enhancement** and **retrieval-based learning.**

### **Key Features**
- **Query Enhancement**: Improves user queries using LLM-based processing.
- **Generative AI Responses**: Provides accurate answers using White Rabbit.
- **Vector Database Integration**: Retrieves relevant cybersecurity documentation.
- **User-Friendly Frontend**: A Streamlit UI for an intuitive experience.
- **Educational Focus**: Covers fundamental Linux/Bash commands, SSH connectivity, file analysis, and basic cryptography.

### Built With

* [![Python][python-logo]][python-url]
* [![Docker][docker-logo]][docker-url]
* [![FastAPI][fastapi-logo]][fastapi-url]
* [![Streamlit][streamlit-logo]][streamlit-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🏗 **System Architecture**

The system is built on a modular architecture, comprising:
1. **Generative Module API**:
   - Backend for AI model inference and FastAPI endpoints.
2. **Streamlit Module API**:
   - Frontend providing an interactive user interface.

**System Diagram**: Copy and paste the code from `SystemDiagram.md` into [Mermaid Live Editor](https://mermaid.live/) to visualize the architecture.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## ✅ **Prerequisites**

Ensure the following are installed on your system:
- **Docker Engine** (19.03.0+)
- **Docker Compose** (2.0+)
- **NVIDIA GPU** with CUDA support
- **NVIDIA Container Toolkit** (nvidia-docker2)

### 🛠 **Installation Instructions**

<details>
<summary><strong>Installing Docker</strong></summary>

##### Step 1: Uninstall Conflicting Packages
Run the following command to remove any previously installed Docker-related packages that might conflict:
```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
```

##### Step 2: Set Up Docker's `apt` Repository
1. Update the `apt` package index:
    ```bash
    sudo apt-get update
    ```

2. Install required packages:
    ```bash
    sudo apt-get install ca-certificates curl
    ```

3. Create the `/etc/apt/keyrings` directory:
    ```bash
    sudo install -m 0755 -d /etc/apt/keyrings
    ```

4. Download and add Docker’s GPG key:
    ```bash
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    ```

5. Add the Docker repository:
    ```bash
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    ```

6. Update the package index:
    ```bash
    sudo apt-get update
    ```

##### Step 3: Install Docker Packages
Install Docker Engine, CLI, and associated plugins:
```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

##### Step 4: Verify the Installation
Run the following command to verify that Docker is installed correctly:
```bash
sudo docker run hello-world
```

##### Step 5: Enable Non-Root Docker Usage
1. Add your user to the Docker group:
    ```bash
    sudo groupadd docker
    sudo usermod -aG docker $USER
    ```

2. Apply the changes:
    ```bash
    newgrp docker
    ```

3. Verify that you can run Docker commands without `sudo`:
    ```bash
    docker run hello-world
    ```

##### Step 6: Enable Docker on Boot
To start Docker and containerd on system boot:
```bash
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---
</details>

<details>
<summary><strong>Installing NVIDIA Container Toolkit</strong></summary>

##### Step 1: Add NVIDIA’s Package Repository
Run the following command to add NVIDIA's GPG key and repository:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

##### Step 2: Enable the Experimental Components
Edit the repository file to enable experimental components:
```bash
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

##### Step 3: Update Package Index
```bash
sudo apt-get update
```

##### Step 4: Install NVIDIA Container Toolkit
```bash
sudo apt-get install -y nvidia-container-toolkit
```

##### Step 5: Configure the Runtime
Set NVIDIA as the default runtime for Docker:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

##### Step 6: Verify the Installation
1. Check that the NVIDIA Container Toolkit is installed:
    ```bash
    dpkg -l | grep nvidia-container-toolkit
    ```
    **Expected Output**:
    ```plaintext
    ii  nvidia-container-toolkit                   1.16.2-1                                amd64        NVIDIA Container toolkit
    ii  nvidia-container-toolkit-base              1.16.2-1                                amd64        NVIDIA Container Toolkit Base
    ```

2. Verify NVIDIA runtime is available to Docker:
    ```bash
    sudo docker info | grep Runtimes
    ```
    **Expected Output**:
    ```plaintext
    Runtimes: io.containerd.runc.v2 nvidia runc
    ```
</details>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## ⚙️ **Setup Instructions**

### 1. Clone the Repository
```bash
git clone https://github.com/iamamiramine/generative-cybersecurity.git
cd generative-cybersecurity
```

### 2. Install Docker and NVIDIA Toolkit
Refer to [this section in the README](#prerequisites) for the installation commands.

### 3. Create Docker Network
```bash
docker network create generative-cybersecurity-network
```

### 4. Prepare Directory Structure
Ensure the following folders exist:
- `models/` - Stores AI models.
- `data/` - Stores application data (only `.txt` files are supported).

### 5. Download the Model
Use the following command to download the required model:
```bash
curl -X POST http://localhost:7575/download_llm -H "Content-Type: application/json" -d '{"model_name": "WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B"}'
```

Alternatively, use Swagger UI at `http://localhost:7575/docs#/`.

### 6. Build and Run the Application
```bash
docker compose build
docker compose up
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🚀 **Usage**

### **Accessing the Application**
- **Streamlit Frontend**: [http://localhost:7500](http://localhost:7500)
- **Generative API Swagger UI**: [http://localhost:7575/docs](http://localhost:7575/docs)

### **FastAPI Endpoints**
1. **Without Context**:
   - `POST /load_model`
   - `POST /load_pipeline`
   - `POST /load_chain`
   - `POST /generate`
2. **With Context**:
   - `POST /load_model`
   - `POST /load_pipeline`
   - `POST /load_docs`
   - `POST /load_ensemble_retriever_from_docs`
   - `POST /load_chain`
   - `POST /generate`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 📂 **Directory Structure**

```plaintext
generative-cybersecurity/
├── models/                    # Contains AI models
├── data/                      # Contains application data
├── shared/                    # Shared resources
│   ├── assets/                # Images and other assets
│   └── config/                # Configuration files
├── generative-module-api/     # Backend service
└── streamlit-module-api/      # Frontend service
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🛠 **Streamlit UI Workflow**

1. Load the model.
2. Load the pipeline.
3. Load the documents.
4. Load the ensemble retriever.
5. Load the chain.
6. Generate text.

> The UI dynamically reloads components like the pipeline and chain when parameters or context change.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Top contributors:

<a href="https://github.com/iamamiramine/generative-cybersecuritye/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=iamamiramine/generative-cybersecurity" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 💡 **Acknowledgments**

Special thanks to:
- The **White Rabbit** open-source community for their LLM contributions.
- The **OverTheWire** community for providing ethical cybersecurity challenges.
- Open-source tools and libraries that made this project possible.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/iamamiramine/generative-cybersecurity/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/iamamiramine/generative-cybersecurity/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/iamamiramine/generative-cybersecurity/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/iamamiramine/generative-cybersecurity/issues
[python-logo]: https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[docker-logo]: https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white
[docker-url]: https://www.docker.com/
[fastapi-logo]: https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white
[fastapi-url]: https://fastapi.tiangolo.com/
[streamlit-logo]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white
[streamlit-url]: https://streamlit.io/
