FROM python:3.11-slim

WORKDIR /Home

# System deps (for FAISS, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /Home/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /Home

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

CMD ["streamlit", "run", "Home.py"]
