# Makefile for Boston Weather Analysis Project
# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Format code 
format:
	black bos_temp.py test_bos_temp.py

# Run tests 
test:
	python -W ignore::DeprecationWarning test_bos_temp.py

# Run tests with pytest 
test-pytest:
	python -m pytest test_bos_temp.py -v

# Run the main analysis 
run:
	python bos_temp.py

# Lint code (using your actual files)
lint:
	flake8 bos_temp.py test_bos_temp.py --max-line-length=88 --extend-ignore=E203,W503

# Clean temporary files
clean:
	rm -rf __pycache__/
	rm -f *.pyc
	rm -f test_pred.png
