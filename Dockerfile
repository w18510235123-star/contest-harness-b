FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONTEST_SUBMISSION=contest_submission

WORKDIR /opt/contest-harness
COPY sdk/pyproject.toml /opt/contest-harness/pyproject.toml
COPY sdk/contest_harness /opt/contest-harness/contest_harness
RUN python3 -m pip install --no-cache-dir /opt/contest-harness \
    && python3 -m pip install --no-cache-dir numpy==1.26.4

WORKDIR /app
COPY harness.py contest_submission.py /app/
COPY baselines /app/baselines

USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]
