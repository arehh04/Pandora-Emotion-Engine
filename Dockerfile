FROM python:3.12-slim

# Install system dependencies (NGINX, Redis, Supervisord, OpenMP for XGBoost)
RUN apt-get update && apt-get install -y \
    nginx \
    redis-server \
    supervisor \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy all project files
COPY . .

# Setup NGINX
RUN rm /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/sites-available/default
RUN ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default

# Setup Supervisord
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Give appropriate permissions for Hugging Face Spaces (runs as non-root user 'user')
RUN useradd -m -u 1000 user && \
    mkdir -p /var/log/supervisor /var/log/nginx /var/lib/nginx /var/run/nginx /run /var/lib/redis /var/log/redis && \
    chown -R user:user /app /var/log/supervisor /var/log/nginx /var/lib/nginx /var/run/nginx /run /etc/nginx /var/lib/redis /var/log/redis

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Expose the standard HF port
EXPOSE 7860

# Run supervisor to manage Redis, FastAPI, and NGINX
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
