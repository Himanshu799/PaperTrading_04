# ─── 1) Builder: install deps ─────────────────────────────────────────────────
FROM python:3.10-slim AS builder

ENV \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# install system build‐essentials (for any packages with C extensions)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# copy & install only Python requirements
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


# ─── 2) Final: copy runtime bits only ──────────────────────────────────────────
FROM python:3.10-slim

ENV \
    PYTHONUNBUFFERED=1 \
    PATH=/install/bin:$PATH \
    PYTHONPATH=/install/lib/python3.10/site-packages

WORKDIR /app

# copy installed packages from builder
COPY --from=builder /install /install

# copy your application code
COPY deploy.py .

# if you mount or download models at runtime, skip copying them here
# otherwise, uncomment and selectively copy only what you absolutely need:
# COPY models/small_model.h5 models/
# COPY processed_data/aapl_state_cnnlstm_test.npy processed_data/

# run
CMD ["python", "deploy.py"]
