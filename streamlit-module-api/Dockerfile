FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY ./app.py ./app.py
COPY .streamlit /app/.streamlit

ENTRYPOINT ["streamlit", "run", "--server.port=8501", "--server.address=0.0.0.0"]

CMD ["app.py"]