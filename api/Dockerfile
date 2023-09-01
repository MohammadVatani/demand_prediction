FROM python:3.8

ENV POETRY_VERSION = '1.5.1'

RUN apt-get update && apt-get install -y curl

RUN curl -sSL https://install.python-poetry.org | python3 - --version '$POETRY_VERSION'

ENV PATH="${PATH}:/root/.poetry/bin"

WORKDIR /app

COPY ./pyproject.toml ./

RUN poetry add --without dev

COPY . .

CMD ["python", "-m", "uvicorn", "main:app"]
