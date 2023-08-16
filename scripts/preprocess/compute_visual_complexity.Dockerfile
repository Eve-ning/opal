FROM python:3.9

# Set the working directory to /app
WORKDIR /app
RUN pip3 install pandas mysql-connector-python reamber tqdm SQLAlchemy
COPY ./compute_visual_complexity.py /app/compute_visual_complexity.py

CMD ["python3", "-m", "compute_visual_complexity"]