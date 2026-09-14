FROM python:3.11-slim

WORKDIR /app


COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY api.py graph.py generator.py grader.py retriever.py llm_utils.py \
     rate_limiter.py logging_config.py ./
COPY static/ ./static/

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]