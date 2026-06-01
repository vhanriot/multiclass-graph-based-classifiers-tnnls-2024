FROM python:3.12-slim

WORKDIR /main

RUN apt-get update && apt-get install -y --no-install-recommends

VOLUME ["/results", "/figures"]

COPY requirements.txt .
COPY pyproject.toml .

RUN pip install --no-cache-dir -r requirements.txt

COPY results_bkp/ /main/results_bkp
COPY data/ /main/data
COPY src/ /main/src
COPY scripts/ /main/scripts

RUN pip install -e .

CMD ["python"]
