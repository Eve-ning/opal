FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN pip3 install --upgrade pip
RUN pip3 install pytorch-lightning scikit-learn pandas
RUN pip3 install tensorboard

COPY ./opal ./opal
COPY ./entrypoint.sh /entrypoint.sh

CMD ["/bin/bash", "/entrypoint.sh"]
