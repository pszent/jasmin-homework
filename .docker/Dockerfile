FROM ubuntu:latest
RUN apt-get update -y && apt-get install -y python3-venv python3-pip libgtk2.0-dev
COPY . /home/jasmin-demo/
WORKDIR /home/jasmin-demo/
RUN pip3 install -r .docker/requirements.txt
RUN python3 setup.py install
CMD ['python3 /home/jasmin-demo/app/classification.py']
