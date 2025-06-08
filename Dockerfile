FROM python:3.10-slim

RUN apt-get update -y \ && 
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev \ &&
    rm -rf /var/lib/apt/lists/*

# Установка python-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект в контейнер
COPY . /app
WORKDIR /app

# Открываем порт
EXPOSE 5000

# Запуск приложения
CMD ["python", "app.py"]
