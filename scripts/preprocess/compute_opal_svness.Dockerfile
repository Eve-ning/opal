FROM python:3.9

# Set the working directory to /app
WORKDIR /app
RUN pip3 install pandas mysql-connector-python reamber tqdm SQLAlchemy
COPY ./compute_opal_svness.py /app/compute_opal_svness.py

CMD ["python3", "-m", "compute_opal_svness"]