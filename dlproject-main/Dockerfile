# Use an official Python image
FROM python:3.10-slim

# Set environment variables
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential libssl-dev libffi-dev python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app
WORKDIR /app

# Install Python dependencies in the virtual environment
RUN $VIRTUAL_ENV/bin/pip install --upgrade pip
RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir -r requirements.txt

# Expose the application port (if required)
EXPOSE 5000 

# Command to run the application
CMD ["python", "app.py"]  # Change this if your entry point is different
