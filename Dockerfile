FROM python:3

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir gym torch

ENV WEIGHTS_OUTPUT_FILE /opt/ml/model

COPY . .

ENTRYPOINT ["python3", "./first.py"]
