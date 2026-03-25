FROM quay.io/jupyter/base-notebook:python-3.12

WORKDIR /home/jovyan/work 

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt  