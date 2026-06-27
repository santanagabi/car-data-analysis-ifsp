# Imagem para o Trabalho Final GETI — JupyterLab + stack de Data Science.
# Build reprodutível: versões travadas em requirements.txt (pandas 3.0.3, sklearn 1.9.0, etc.).
FROM python:3.13-slim

# libgomp1 é necessário para o OpenMP usado por scikit-learn (KMeans/NearestNeighbors).
# build-essential cobre a compilação de eventuais wheels sem distribuição binária.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Instala as dependências primeiro para aproveitar o cache de camadas do Docker.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8888

# JupyterLab sem token (acesso local via http://localhost:8888/lab).
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.token=", \
     "--ServerApp.password=", \
     "--ServerApp.root_dir=/work"]
