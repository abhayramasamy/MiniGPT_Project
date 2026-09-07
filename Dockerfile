FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py model.py build.py load_weights.py tokenizer.py inference.py stats.py app.py ./
COPY models/ ./models/

EXPOSE 5000

CMD ["python", "app.py"]