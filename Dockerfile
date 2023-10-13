FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org

COPY grobid_magneto/ grobid_magneto
COPY commons/ commons
COPY resources/nims_proxy.cer .
COPY tiktoken_cache ./tiktoken_cache
COPY grobid_magneto/document_qa/.streamlit ./.streamlit

# extract version
COPY .git ./.git
RUN git rev-parse --short HEAD > revision.txt
RUN rm -rf ./.git

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV PYTHONPATH "${PYTHONPATH}:."

ENTRYPOINT ["streamlit", "run", "document_qa/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
