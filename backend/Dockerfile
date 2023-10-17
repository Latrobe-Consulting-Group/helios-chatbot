# Use an official Python runtime as a parent image
FROM python:3.11.5-slim-bullseye

# Add metadata labels for better visibility
LABEL maintainer="jack@latrobe.group"

ARG VERSION="0.0.3"
ARG OPENAIKEY=""
ENV VERSION=$VERSION

# Setting user to non-root for added security
RUN useradd -ms /bin/bash appuser
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV VERSION=$VERSION
ENV LANCEDB_CONFIG_DIR="/helios/"
ENV OPENAI_CHAT_MODEL="gpt-3.5-turbo-0613"
ENV OPENAI_TEMPERATURE=0.1
ENV OPENAI_MAX_TOKENS=300
ENV HELIOS_URL="helios.latrobe.group"
ENV OPENAI_API_KEY=$OPENAIKEY

# Create and set working directory
WORKDIR /helios

# Copy requirements file and install packages
# Doing this separately prevents re-installing dependencies if code changes
COPY requirements.txt /helios/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /helios/

# Expose port 8000 for the REST API
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]