.PHONY: help
help:
	@echo "make setup   - creates the virtual env and install packages"
	@echo "make dataset - download the raw dataset"
	@echo "make train   - train a model"
	@echo "make run_api - start web services"

.PHONY: setup
setup:
	pyenv install --skip-existing 3.12.9
	pyenv virtualenvs --bare | grep -e '^celebtwin$$' \
	|| pyenv virtualenv celebtwin
	pyenv local celebtwin
	pip install -r requirements.txt

dataset_zip = pins-face-recognition.zip

.PHONY: dataset
dataset:
	curl --location --continue-at - \
	--output "raw_data/${dataset_zip}" \
	https://www.kaggle.com/api/v1/datasets/download/hereisburak/pins-face-recognition
	cd raw_data && unzip "${dataset_zip}"


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

.PHONY: cleans
clean:
	rm -fr **/__pycache__ **/*.pyc

.PHONY: run_api
run_api:
	uvicorn celebtwin.api.fast:app --reload

.PHONY: train
train:
	python -m celebtwin train
