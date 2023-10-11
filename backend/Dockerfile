# Use an official Python runtime as a parent image
FROM python:3.11.5-slim-bullseye

# Add metadata labels for better visibility
LABEL maintainer="jack@latrobe.group"
LABEL version="0.0.1"

# Setting user to non-root for added security
RUN useradd -ms /bin/bash appuser
USER appuser

# Set environment variable for Python unbuffered to ensure logs can be streamed
ENV PYTHONUNBUFFERED=1
ENV VERSION="0.0.2"

# Create and set working directory
WORKDIR /app

# Copy requirements file and install packages
# Doing this separately prevents re-installing dependencies if code changes
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Expose port 8000 for the REST API
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]