FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app

WORKDIR /app

RUN addgroup --system app \
    && adduser --system --ingroup app --home /home/app app \
    && mkdir -p /home/app \
    && chown app:app /home/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .
RUN mkdir -p /app/static/uploads /app/inference_results /app/data \
    && chown -R app:app /app/static/uploads /app/inference_results /app/data

USER app

EXPOSE 5000

CMD ["gunicorn", "--config", "gunicorn_config.py", "app1:app"]
