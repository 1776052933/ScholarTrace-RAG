FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt ./
COPY Auto_Paper_UI/requirements-ui.txt Auto_Paper_UI/requirements-ui.txt
RUN --mount=type=cache,id=scholartrace-pip-m5b,target=/root/.cache/pip pip install -r requirements.txt -r Auto_Paper_UI/requirements-ui.txt
COPY . .
