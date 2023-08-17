FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app/

COPY ./opal /app/opal

RUN pip3 install --upgrade pip
RUN pip3 install pytorch-lightning[extra]

CMD ["python3", "-m", "opal.train"]
