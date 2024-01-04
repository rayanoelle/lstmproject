FROM python:3.8-bullseye
RUN pip install --upgrade pip
COPY ./autoscaler/requirements.txt .
RUN pip3 install -r requirements.txt
COPY ./models/cpu_util/cnnBilstm/ /models/cpu_util/cnnBilstm/
COPY ./data/barycenters/ /data/barycenters/
COPY ./autoscaler /app
WORKDIR /app
CMD ["python3", "requesthandler.py"]