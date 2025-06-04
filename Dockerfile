FROM python:3.13-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies for pyaudio and other common tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libportaudio2 \
    portaudio19-dev \
    ffmpeg \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY main.py .

# Expose the port the app runs on (Render will map this)
# Render typically sets a PORT env var, often 10000.
# Your app should listen on $PORT or a default like 8000.
EXPOSE 8000

# Command to run the application
# Ensure main.py is configured to listen on 0.0.0.0 and use $PORT
CMD ["python", "main.py"] 