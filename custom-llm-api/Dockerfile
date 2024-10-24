FROM python:3.12-slim-bullseye

WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN pip install --upgrade -r requirements.txt

COPY ./app /app

EXPOSE 80

#can ignore deprecation warnings as long as there are no errors
# RUN pytest -W ignore::DeprecationWarning

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]