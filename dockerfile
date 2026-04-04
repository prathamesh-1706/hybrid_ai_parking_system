# Use official Python 3.10 base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Upgrade pip first
RUN pip install --upgrade pip

# Install dependencies (excluding torch/numpy for now)
RUN pip install --no-cache-dir pandas opencv-python gymnasium matplotlib seaborn fastapi uvicorn

# Install compatible NumPy and PyTorch versions
RUN pip install --no-cache-dir "numpy<2" torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# If you have a requirements.txt, install any other dependencies without overwriting numpy
# RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7860
EXPOSE 7860

# Start the FastAPI app automatically when the container runs
# Replace 'api:app' with your filename and FastAPI app object if different
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]