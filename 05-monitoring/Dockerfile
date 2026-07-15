FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

# dependency files live in the repo root (build context is ..)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked

# only the monitoring app code into the image
COPY 05-monitoring/ ./

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
