import os


def positive_int(name, default):
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


bind = "0.0.0.0:5000"
workers = positive_int("GUNICORN_WORKERS", 1)
threads = positive_int("GUNICORN_THREADS", 4)
timeout = positive_int(
    "GUNICORN_TIMEOUT",
    positive_int("MEDGEMMA_API_TIMEOUT", 600) + 60,
)
accesslog = "-"
errorlog = "-"
