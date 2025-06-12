FROM python:3.10.17-bookworm
WORKDIR /app

# libgl1-mesa-glx: Fix ImportError: libGL.so.1: cannot open shared object file
# cmake: To build annoy from source.
RUN apt-get update
RUN apt install -y libgl1-mesa-glx cmake

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN CPPFLAGS="-DNO_MANUAL_VECTORIZATION" pip install --no-binary annoy annoy==1.17.3
RUN pip install --no-cache-dir -r requirements.txt
COPY celebtwin /app/celebtwin/
COPY training_outputs/annoy/skip-Facenet-Facenet2018-100-euclidean \
/app/training_outputs/annoy/skip-Facenet-Facenet2018-100-euclidean

CMD ["sh", "-c", "exec uvicorn --host 0.0.0.0 --port $PORT celebtwin.api.fast:app"]
