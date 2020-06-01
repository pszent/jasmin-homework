FROM ubuntu:latest
RUN apt-get update -y && apt-get install -y python3-venv python3-pip
COPY .docker/requirements.txt .
RUN pip3 install -r requirements.txt
