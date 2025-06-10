FROM python:3.10.17-bookworm
WORKDIR /app

# Fix ImportError: libGL.so.1: cannot open shared object file
RUN apt-get update
RUN apt install -y libgl1-mesa-glx

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY celebtwin /app/celebtwin/
CMD uvicorn --host 0.0.0.0 --port $PORT celebtwin.api.fast:app
