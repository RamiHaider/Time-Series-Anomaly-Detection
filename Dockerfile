FROM ultralytics/ultralytics:latest

WORKDIR /app/

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/

CMD ["python", "main.py"]
