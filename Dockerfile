FROM python:3.10.12

#set working directory
WORKDIR /pytorch_classifier

#copy requirements into container
COPY requirements.txt requirements.txt

#install dependencies
RUN pip install -r requirements.txt

