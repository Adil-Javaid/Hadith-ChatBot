FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000  # Expose port 5000 for the Flask app

CMD ["python", "Adil(javaidadil835@gmail.com).py"]
