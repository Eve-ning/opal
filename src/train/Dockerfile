FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN pip3 install --upgrade pip
RUN pip3 install pytorch-lightning[extra] scikit-learn pandas

COPY ./opal ./opal
COPY ./entrypoint.sh /entrypoint.sh

CMD ["python3", "-m", "opal.train"]
