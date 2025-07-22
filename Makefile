# Error on undefined variables, or if any command in a pipe fails.
SHELL := bash -c
.SHELLFLAGS := -u -o pipefail

.DELETE_ON_ERROR:  # Delete target if its recipe errors out
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

.PHONY: help
help:
	@echo "make setup          - creates the virtual env and install packages"
	@echo "make dataset        - download the raw dataset"
	@echo "make clean          - Remove image and cache"
	@echo "make train          - train a model"
	@echo "make run_api        - start web services"
	@echo "make lint           - run code analysis and style checks"
	@echo "make format         - automatically format code"
	@echo "make requirements   - install requirements"
	@echo "make pip-compile    - compile requirements files"
	@echo "make image          - build the Docker image"
	@echo "make image-run      - run the Docker image"
	@echo "make deploy        - instruction to deploy the code to production"
	@echo "make image-on-instance - build the Docker image on the instance"
	@echo "make image-create-repo - create the Docker repository"
	@echo "make image-auth     - authenticate to the Docker repository"
	@echo "make image-push     - push the Docker image to the repository"
	@echo "make image-deploy   - deploy the Docker image to Google Cloud Run"
	@echo "make start-instance - start up the instance"
	@echo "make stop-instance  - stop the instance"
	@echo "make ssh            - ssh into the instance"

.PHONY: setup
setup:
	uv venv
	uv pip install --upgrade pip
	uv pip install -r requirements.txt -r requirements-dev.txt

dataset_zip = pins-face-recognition.zip

.PHONY: dataset
dataset:
	curl --location --continue-at - \
	--output "raw_data/${dataset_zip}" \
	https://www.kaggle.com/api/v1/datasets/download/hereisburak/pins-face-recognition
	cd raw_data && unzip -q "${dataset_zip}"

.PHONY: clean
clean:
	shopt -s globstar && rm -fr .venv dockerbuild **/__pycache__ **/*.pyc *.egg-info

.PHONY: run_api
run_api:
	uv run uvicorn celebtwin.api:app --reload

.PHONY: train
train:
	python -m celebtwin train --dataset aligned --model weekend --classes 5 \
	--learning-rate 0.0001

.PHONY: lint
lint:
	-uv run ruff check celebtwin
	-uv run mypy celebtwin
	-uv run isort --check celebtwin

.PHONY: format
format:
	-uv run isort celebtwin

.PHONY: requirements
requirements: pip-compile
	pip install --quiet --upgrade pip
	pip install --quiet -r requirements.txt -r requirements-dev.txt

.PHONY: pip-compile
pip-compile: requirements.txt requirements-dev.txt

PIP_COMPILE_FLAGS = --strip-extras --generate-hashes
requirements.txt: pyproject.toml
	uv pip compile $(PIP_COMPILE_FLAGS) $< > $@

# Extract dev dependencies from pyproject.toml
EXTRACT_DEV_REQS = uv run python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['dependency-groups']['dev']))"

requirements-dev.txt: pyproject.toml requirements.txt
	$(EXTRACT_DEV_REQS) > requirements-dev.in
	uv pip compile $(PIP_COMPILE_FLAGS) --constraint=requirements.txt requirements-dev.in > $@
	rm requirements-dev.in

# Name of Docker image
IMAGE=celebtwin

weights_file_names = facenet_weights.h5 vgg_face_weights.h5
weights_files = \
$(addprefix dockerbuild/.deepface/weights/, $(weights_file_names))
weights_url_base = \
https://github.com/serengil/deepface_models/releases/download/v1.0

.PHONY: image
# Set NO_CACHE=--no-cache to invalidate the cache.
# Set PLATFORM='--platform linux/amd64' to build for x86_64 architecture.
image: $(weights_files)
	$(MAKE) $(weights_files)
	# Serialize build process to avoid lock conflict on apt cache
	docker --debug build $(PLATFORM) $(NO_CACHE) --target build -t $(IMAGE) .
	docker --debug build $(PLATFORM) -t $(IMAGE) .

$(weights_files):
	mkdir -p $(dir $@)
	curl --location --output $@.part $(weights_url_base)/$(notdir $@)
	mv $@.part $@

.PHONY: image-run
image-run:
	docker run -e PORT=8000 -p 8000:8000 $(IMAGE)

.PHONY: deploy
deploy:
	@echo "How to build image for production:"
	@echo "$$ git commit"
	@echo "$$ git push  # Must be in main branch, upload changes to GitHub"
	@echo "$$ make start-instance"
	@echo "$$ make image-on-instance"
	@echo "$$ make image-deploy"
	@echo "$$ make stop-instance"

.PHONY: image-on-instance
image-on-instance:
	$(MAKE) ssh COMMAND='cd celebtwin && git pull origin main && make PLATFORM="--platform linux/amd64" image image-push'

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
		--memory=16G \
		--concurrency=16 \
		--allow-unauthenticated

.PHONY: start-instance
start-instance:
	gcloud compute instances start --zone=$(ZONE) $(INSTANCE)

.PHONY: stop-instance
stop-instance:
	gcloud compute instances stop --zone=$(ZONE) $(INSTANCE)

COMMAND =
.PHONY: ssh
ssh:
	gcloud compute ssh $(INSTANCE) --project=$(PROJECT) --zone=$(ZONE) -- -A $(COMMAND)
