FROM ubuntu:latest
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y python3-venv python3-pip
RUN pip3 install notebook
COPY . /home/jasmin-demo/
WORKDIR /home/jasmin-demo/
RUN pip3 install -r .docker/requirements.txt
RUN python3 setup.py install
WORKDIR /home/jasmin-demo/app
CMD ["python3", "./classification.py"]