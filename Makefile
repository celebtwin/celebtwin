.PHONY: help
help:
	echo "make setup   - creates the virtual env and install packages"
	echo "make dataset - download the raw dataset"

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
GS_DIR=gs://datascience-mlops/taxi-fare-ny

show_sources_all:
	-ls -l ${ML_DIR} | wc -l
#	-bq ls ${BQ_DATASET}
#	-gsutil ls gs://${BUCKET_NAME}

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p ${ML_DIR}

clean:
	@rm -fr **/__pycache__ **/*.pyc
