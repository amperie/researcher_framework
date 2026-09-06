FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/researcher-venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /workspace/NeuralSignalResearcher

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY configs ./configs
COPY core ./core
COPY web ./web
COPY main.py run_campaign.py run_node.py rebuild_neo4j_from_mongo.py ./

RUN uv sync --frozen

CMD ["uv", "run", "python", "-m", "core.plugins.trading.ui_agent", "idea-batch"]
