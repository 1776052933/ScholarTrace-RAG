FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt ./
COPY Auto_Paper_UI/requirements-ui.txt Auto_Paper_UI/requirements-ui.txt
RUN pip install --no-cache-dir -r requirements.txt -r Auto_Paper_UI/requirements-ui.txt
COPY . .
