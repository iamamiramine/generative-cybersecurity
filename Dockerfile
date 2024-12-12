FROM kalilinux/kali-rolling

RUN apt update && apt -y install python3 python3-pip curl wget gcc build-essential git gh

# install conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
  /bin/bash ~/miniconda.sh -b -p /opt/conda

# create env with python 3.5
RUN /opt/conda/bin/conda create -y -n myenv python=3.11

WORKDIR /app

ENV PATH=/opt/conda/envs/myenv/bin:$PATH

RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# RUN /opt/conda/bin/conda install nvidia/label/cuda-12.4.0::cuda-toolkit -y
RUN /opt/conda/bin/conda install -c conda-forge cudatoolkit -y

RUN git clone https://github.com/oobabooga/text-generation-webui.git

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install packages from apt
RUN apt-get install -y nmap

COPY ./src ./src
COPY ./app.py ./app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]