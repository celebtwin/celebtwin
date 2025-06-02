help:
	"make setup - creates the virtual env and install packages"

setup:
	pyenv install --skip-existing 3.12.9
	pyenv virtualenvs --bare | grep -e '^celebtwin$$' \
	|| pyenv virtualenv celebtwin
	pip install -r requirements.txt
