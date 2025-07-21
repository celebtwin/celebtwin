# Base image for building
FROM python:3.12-slim-bookworm AS build
# FROM python:3.12-slim-bookworm
WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt \
  apt update && apt-get --no-install-recommends install --yes g++

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set up virtual environment
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY README.md pyproject.toml uv.lock /app/
RUN uv venv

COPY celebtwin /app/celebtwin
RUN --mount=type=cache,target=/root/.cache/uv \
--mount=type=cache,target=/root/.cache/pip \
uv pip install --upgrade pip && \
# Install annoy with custom build flags
CPPFLAGS="-DNO_MANUAL_VECTORIZATION" \
pip install --no-binary all annoy==1.17.3 && \
uv pip install .

# Final image
FROM python:3.12-slim-bookworm
WORKDIR /app

# libgl1-mesa-glx: Fix ImportError: libGL.so.1: cannot open shared object file
RUN --mount=type=cache,target=/var/cache/apt \
  apt update && apt install --yes libgl1-mesa-glx libglib2.0-0

COPY --from=build /usr/local /usr/local
COPY --from=build /app/ /app/

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY training_outputs/ann/builtin-VGG-Face-VGGFace2-brute \
/app/training_outputs/ann/builtin-VGG-Face-VGGFace2-brute
COPY training_outputs/ann/builtin-Facenet-Facenet2018-brute \
/app/training_outputs/ann/builtin-Facenet-Facenet2018-brute

# Copy deepface models into the image. The files are downloaded in Makefile.
COPY dockerbuild/.deepface /app/.deepface/

# Tells Deepface where model files are. Also where it will download if needed.
ENV DEEPFACE_HOME=/app

CMD ["sh", "-c", "exec uvicorn --host 0.0.0.0 --port $PORT celebtwin.api:app"]
