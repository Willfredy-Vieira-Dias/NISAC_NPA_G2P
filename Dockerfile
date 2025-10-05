# Base image (usa Python 3.11 por estabilidade e compatibilidade com Google libs)
FROM python:3.11-slim

# Expõe a porta da aplicação
EXPOSE 8000

# Impede a criação de arquivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1

# Desativa buffer do Python para logs em tempo real
ENV PYTHONUNBUFFERED=1

# Define diretório de trabalho
WORKDIR /app

# Copia apenas o requirements primeiro (para aproveitar cache de build)
COPY requirements.txt .

# Instala dependências de sistema e Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && python -m pip install --no-cache-dir -r requirements.txt \
 && python -m pip install --no-cache-dir google --upgrade \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copia o código do projeto
COPY . /app

# Cria um usuário não-root por segurança
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Comando de inicialização
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "gemini_integracao_api_npa:app"]
