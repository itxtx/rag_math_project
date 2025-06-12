# /Dockerfile

# Start from a base image with Python
FROM python:3.10-slim-bookworm

# Set an environment variable to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    perl \
    cpanminus \
    pkg-config \ 
    libxml2-dev \
    libxslt1-dev \  
    zlib1g-dev \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-science \
    texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

# Use Perl's package manager (cpanm) to install LaTeXML
RUN cpanm --notest LaTeXML

# Set up the working directory in the container
WORKDIR /app

# 4. Install uv, your preferred Python package manager
RUN pip install uv

# 5. Copy the pyproject.toml file and install dependencies with uv
# This uses pyproject.toml as the source of truth, not requirements.txt
COPY pyproject.toml .
RUN uv pip install . --system

# 6. Copy the rest of your application source code into the container
COPY . .
# Expose the port your FastAPI application runs on (assuming it's 8000)
EXPOSE 8000

# The command to run your application
# This will start the FastAPI server from your main_api.py
CMD ["uvicorn", "src.api.main_api:app", "--host", "0.0.0.0", "--port", "8000"]