FROM python:3.10

WORKDIR /app

COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt \
    -i https://pypi.org/simple

# Install light packages first
RUN pip3 install --no-cache-dir -r requirements-base.txt

# Install heavy packages separately (can be retried)
RUN pip3 install --no-cache-dir -r requirements-heavy.txt

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]