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
COPY training_outputs/ann/builtin-VGG-Face-VGGFace2-brute \
/app/training_outputs/ann/builtin-VGG-Face-VGGFace2-brute
COPY training_outputs/ann/builtin-Facenet-Facenet2018-brute \
/app/training_outputs/ann/builtin-Facenet-Facenet2018-brute

# Copy deepface models into the image. The files are downloaded in Makefile.
COPY dockerbuild/.deepface /app/.deepface/
# Tells Deepface where model files are. Also where it will download if needed.
ENV DEEPFACE_HOME=/app

CMD ["sh", "-c", "exec uvicorn --host 0.0.0.0 --port $PORT celebtwin.api:app"]
