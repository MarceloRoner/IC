# ---------------- imagem base mínima ----------------
    FROM python:3.11-slim

    # ----- dependências de sistema para WeasyPrint -----
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            libpango-1.0-0 libcairo2 libgdk-pixbuf2.0-0 && \
        apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # ----- dependências Python -----
    COPY requirements.txt /tmp/req.txt
    RUN pip install --no-cache-dir -r /tmp/req.txt
    
    # (garanta que requirements.txt contém weasyprint==61.0 e gunicorn)
    
    # ----- cópia do código -----
    WORKDIR /app
    COPY . /app
    
    # ----- porta fornecida pelo Render -----
    ENV PORT 8000
    EXPOSE 8000
    
    # ----- comando de start -----
    CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:${PORT}"]
    