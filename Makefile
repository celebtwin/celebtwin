.PHONY: help
help:
	@echo "make setup   - creates the virtual env and install packages"
	@echo "make dataset - download the raw dataset"
	@echo "make train   - train a model"
	@echo "make run_api - start web services"
	@echo "make lint    - run code analysis and style checks"
	@echo "make requirements - install requirements"
	@echo "make pip-compile  - compile requirements files"
	@echo "make ssh       - ssh into the instance"

PY_VERSION=3.10.17
.PHONY: setup
setup:
	pyenv install --skip-existing $(PY_VERSION)
	pyenv virtualenvs --bare | grep -e '^celebtwin$$' \
	|| pyenv virtualenv $(PY_VERSION) celebtwin
	pyenv local celebtwin
	$(MAKE) requirements

dataset_zip = pins-face-recognition.zip

.PHONY: dataset
dataset:
	curl --location --continue-at - \
	--output "raw_data/${dataset_zip}" \
	https://www.kaggle.com/api/v1/datasets/download/hereisburak/pins-face-recognition
	cd raw_data && unzip -q "${dataset_zip}"


ML_DIR=./raw_data/preprocessed
HTTPS_DIR=https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/

.PHONY: show_sources_all
show_sources_all:
	-ls -l ${ML_DIR} | wc -l
#	-bq ls ${BQ_DATASET}
#	-gsutil ls gs://${BUCKET_NAME}

.PHONY: reset_local_files
reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p ${ML_DIR}

.PHONY: clean
clean:
	rm -fr **/__pycache__ **/*.pyc dockerbuild *.egg-info

.PHONY: run_api
run_api:
	uvicorn celebtwin.api.fast:app --reload

.PHONY: train
train:
	python -m celebtwin train --dataset aligned --model weekend --classes 5 \
	--learning-rate 0.0001

.PHONY: lint
lint:
	-ruff check celebtwin
	-mypy celebtwin

.PHONY: requirements
requirements: pip-compile
	pip install --quiet --upgrade pip
	pip install --quiet -r requirements.txt -r requirements-dev.txt

.PHONY: pip-compile
pip-compile: requirements.txt requirements-dev.txt

requirements.txt: pyproject.toml
	pip-compile --quiet --strip-extras pyproject.toml

requirements-dev.txt: requirements-dev.in requirements.txt
	pip-compile --quiet --strip-extras --constraint=requirements.txt \
	requirements-dev.in

# Name of Docker image
IMAGE=celebtwin

.PHONY: image
image: dockerbuild/.deepface/weights/facenet_weights.h5
	docker build -t $(IMAGE) .

dockerbuild/.deepface/weights/facenet_weights.h5:
	mkdir -p $(dir $@)
	curl --location --output $@.tmp https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5
	mv $@.tmp $@

.PHONY: image-run
image-run:
	docker run -e PORT=8000 -p 8000:8000 $(IMAGE)

.PHONY: image-prod
image-prod:
	docker build --platform=linux/amd64 -t $(IMAGE) .

# Google Cloud settings
PROJECT=celebtwin
REGION=europe-west4
REPO=docker
INSTANCE=celebtwin-instance
ZONE=europe-west4-b

.PHONY: image-create-repo
image-create-repo:
	gcloud artifacts repositories create $(REPO) \
		--project=$(PROJECT) \
		--location=$(REGION) \
		--repository-format=docker

.PHONY: image-auth
image-auth:
	gcloud auth configure-docker $(REGION)-docker.pkg.dev

.PHONY: image-push
image-push:
	docker tag $(IMAGE) $(REGION)-docker.pkg.dev/$(PROJECT)/$(REPO)/$(IMAGE)
	docker push $(REGION)-docker.pkg.dev/$(PROJECT)/$(REPO)/$(IMAGE)

.PHONY: image-deploy
image-deploy:
	gcloud run deploy celebtwin-api \
	  --project=$(PROJECT) \
		--region=$(REGION) \
		--image=$(REGION)-docker.pkg.dev/$(PROJECT)/$(REPO)/$(IMAGE) \
		--memory=4G \
		--allow-unauthenticated

.PHONY: ssh
ssh:
	gcloud compute ssh $(INSTANCE) --project=$(PROJECT) --zone=$(ZONE) -- -A
