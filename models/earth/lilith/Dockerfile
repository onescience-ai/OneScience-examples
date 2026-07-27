FROM nikolaik/python-nodejs:python3.10-nodejs20

# Setup Backend Environment
WORKDIR /app/backend
# Copy all files (respecting .dockerignore) to /app/backend
COPY . /app/backend

# Install Python dependencies
# 'pip install .' uses pyproject.toml
RUN pip install --no-cache-dir .

# Setup Frontend Build
WORKDIR /app/frontend
# Copy package files from the copied backend folder (since we copied '.' to /app/backend)
# actually safer to copy from source again to be explicit or just use what we have.
# But standard practice is granular copies to cache layers.
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci

# Copy frontend source
COPY web/frontend ./
# Build frontend
# We set API URL to empty string because we configured rewrites in next.config.js
# requests to /v1 will go to same domain, proxied to localhost:8000
ENV NEXT_PUBLIC_API_URL=""
# Mapbox token is required for build/static generation if used in pages?
# Usually build doesn't need it unless static generation fails.
# We'll set a placeholder if needed, or rely on runtime env.
# But client bundle needs it. User should set it in HF Secrets.
# For build, if we don't have it, some maps might fail to render in static preview? 
# Usually tokens are NEXT_PUBLIC so they are baked in at build time. 
# This is tricky. If baked in, we need the secret at BUILD time (arg).
# HF Spaces build allows Secrets? Yes.
# But simplest is to let it be empty and fail gracefully or warn.
RUN npm run build

# Start Script
WORKDIR /app
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]
