# Gunicorn configuration — auto-loaded from the working directory on startup,
# even when launched as `gunicorn app:app`. Raises the request timeout so the
# credit-card sample (~40s on Render's free tier) isn't killed at the default 30s.
timeout = 180
graceful_timeout = 180
workers = 1
threads = 4
