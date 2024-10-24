FROM python:3.11-slim-bullseye


WORKDIR /dir


COPY ./requirements.txt /dir/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /dir/requirements.txt

COPY ./app /dir/app


CMD ["fastapi", "run", "app/main.py", "--port", "80"]