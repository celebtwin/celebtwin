
.PHONY: help
help:
	echo "make setup   - creates the virtual env and install packages"
	echo "make dataset - download the raw dataset"

.PHONY: setup
setup:
	pyenv install --skip-existing 3.12.9
	pyenv virtualenvs --bare | grep -e '^celebtwin$$' \
	|| pyenv virtualenv celebtwin
	pip install -r requirements.txt

dataset_zip = pins-face-recognition.zip

.PHONY: dataset
dataset:
	curl --location --continue-at - \
	--output "raw_data/${dataset_zip}" \
	https://www.kaggle.com/api/v1/datasets/download/hereisburak/pins-face-recognition
	cd raw_data && unzip "${dataset_zip}"
