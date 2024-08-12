
#FROM python:3.9-slim
FROM python:3.12.4-slim

RUN mkdir /app
RUN mkdir /app/outputs
RUN mkdir /app/torch_install

COPY ./deploy.py /app
COPY ./model.py /app
#COPY ./application_default_credentials.json /app
COPY ./outputs/labels.json /app/outputs
COPY ./torch_install/torch-2.4.0-cp312-cp312-manylinux1_x86_64.whl /app/torch_install

COPY ./requirements.txt ./
RUN pip install --upgrade pip
RUN TMPDIR=/var/tmp pip install /app/torch_install/torch-2.4.0-cp312-cp312-manylinux1_x86_64.whl
RUN pip install -r requirements.txt 

WORKDIR /app

#python3 -m flask --app deploy run --debug
#ENV GOOGLE_APPLICATION_CREDENTIALS=./application_default_credentials.json
ENV FLASK_APP=deploy.py
#ENV FLASK_ENV=development

CMD [ "python3", "-m" , "flask", "run" ]