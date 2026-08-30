FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY README.md ./
COPY src ./src
COPY app.py ./
COPY artifacts ./artifacts
COPY docs/label-rubric-v1.md ./docs/label-rubric-v1.md


RUN python -m pip install --no-cache-dir .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
