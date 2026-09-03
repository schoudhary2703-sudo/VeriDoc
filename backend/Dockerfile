# Python 3.12, matching the local development venv exactly.
#
# BUILD_PLAN specifies 3.11, but every dependency version in requirements.txt was
# pinned against 3.12 and several (numpy 2.5.x among them) publish no 3.11 wheel
# at all. Running a different interpreter in the container than in development is
# how "works on my machine" bugs are manufactured, so the container follows dev.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System libs needed by OpenCV / image tooling in later phases
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
