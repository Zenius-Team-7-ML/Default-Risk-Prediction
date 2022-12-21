FROM python:3.11.1-bullseye

WORKDIR /app

COPY ./requirements.txt .
RUN pip install -r ./requirements.txt

COPY ./rf_model.pkl .
COPY ./main.py/ .

ENTRYPOINT [ "python3", "main.py" ]
