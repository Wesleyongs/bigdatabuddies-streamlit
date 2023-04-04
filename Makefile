# Makefile for managing Python virtual environments

# Set the default target
.DEFAULT_GOAL := help

# Define variables
VENV_NAME := venv
PYTHON_INTERPRETER := python3

# Install dependencies in the virtual environment
install:
	@echo "Creating virtual environment and installing dependencies..."
	$(PYTHON_INTERPRETER) -m venv $(VENV_NAME)
	.\$(VENV_NAME)\Scripts\activate.bat && pip install -r requirements.txt

# Activate the virtual environment
activate:
	@echo "Activating virtual environment..."
	.\$(VENV_NAME)\Scripts\activate.bat

# Deactivate the virtual environment
deactivate:
	@echo "Deactivating virtual environment..."
	deactivate

# Clean up the virtual environment
clean:
	@echo "Cleaning up virtual environment..."
	rm -rf $(VENV_NAME)

run:
	python3.8 -m streamlit run your_app.py

# Display help information
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install   Create virtual environment and install dependencies"
	@echo "  activate  Activate virtual environment"
	@echo "  deactivate  Deactivate virtual environment"
	@echo "  clean     Clean up virtual environment"
	@echo "  help      Display this help information"
