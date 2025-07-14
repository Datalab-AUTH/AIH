FROM python:3.11-slim

EXPOSE 8501

COPY ./streamlit/src /app/src
COPY ./results /results

WORKDIR /app/src

RUN pip install -r requirements.txt

CMD streamlit run Home.py --server.port=8501 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false

