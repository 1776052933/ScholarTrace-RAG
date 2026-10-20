FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
RUN pip install --no-cache-dir torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
COPY requirements-m3-docling.txt ./
COPY Auto_Paper_UI/requirements-ui.txt Auto_Paper_UI/requirements-ui.txt
COPY deploy/constraints-py311.txt deploy/constraints-py311.txt
RUN --mount=type=cache,id=scholartrace-pip-m5b,target=/root/.cache/pip pip install -c deploy/constraints-py311.txt -r requirements.txt -r Auto_Paper_UI/requirements-ui.txt -r requirements-m3-docling.txt && pip check
COPY . .
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.revision=$SOURCE_REVISION
