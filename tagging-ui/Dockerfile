FROM python:3.10-slim
 
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
COPY *.py /app/

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV HOST=0.0.0.0
ENV PORT=80

ENTRYPOINT ["sh", "-c", "streamlit run service.py --server.port=${PORT} --server.address=${HOST}"]