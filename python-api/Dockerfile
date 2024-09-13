FROM python:3.12-alpine

WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN curl -fsSL https://ollama.com/install.sh | sh
RUN ollama pull llama3.1
RUN pip install --upgrade -r requirements.txt

COPY ./app /app

EXPOSE 80

#can ignore deprecation warnings as long as there are no errors
# RUN pytest -W ignore::DeprecationWarning

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]