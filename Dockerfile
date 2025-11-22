# ===========================
# 1. Use supported Python version
# ===========================
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ===========================
# 2. System dependencies
# ===========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# ===========================
# 3. Set workdir
# ===========================
WORKDIR /app

# ===========================
# 4. Install Python dependencies
# ===========================
COPY requirements.txt .

# Force safe versions for PyTorch
RUN pip install --no-cache-dir "numpy<2.0" \
    && pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ===========================
# 5. Copy project code
# ===========================
COPY ./app ./app
COPY ./helper_lib ./helper_lib
COPY ./models ./models
COPY ./gan ./gan
COPY ./artifacts ./artifacts

# ===========================
# 6. Expose port
# ===========================
EXPOSE 8000

# ===========================
# 7. Start server
# ===========================
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

