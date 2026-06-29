FROM python:3.11-slim

# Install system dependencies required for MuJoCo, OpenGL, and X11 forwarding
RUN apt-get update && apt-get install -y \
    build-essential \
    libegl1 \
    libgl1 \
    libosmesa6 \
    libglew-dev \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirement files first to leverage Docker cache
COPY kanellos_progress/requirements.txt ./kanellos_progress/
COPY marios_progress/requirements.txt ./marios_progress/
COPY merged_progress/requirements.txt ./merged_progress/

# Install python dependencies for the merged progress (which has all features)
RUN pip install --no-cache-dir -r merged_progress/requirements.txt

# Copy the rest of the project
COPY . /app

# Set working directory to the merged progress folder
WORKDIR /app/merged_progress

# Default command
CMD ["python", "main.py"]
