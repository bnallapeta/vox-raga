# Phony targets declaration
.PHONY: all build push run clean help acr-build acr-push acr-login

# Variables
IMAGE_NAME ?= vox-raga
TAG ?= 0.0.1
PORT ?= 8888
MODEL_NAME ?= tts_models/en/vctk/vits

# Container runtime configuration
CONTAINER_RUNTIME ?= $(shell which podman 2>/dev/null || which docker 2>/dev/null)
REGISTRY ?= your-registry.azurecr.io
ACR_NAME ?= your-acr-name

# Default target
all: build

# Run tests
test:
	pytest -v tests/

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .pytest_cache

# Build image
build:
	$(CONTAINER_RUNTIME) build -t $(IMAGE_NAME):$(TAG) -f Dockerfile .

# Push image
push:
	$(CONTAINER_RUNTIME) tag $(IMAGE_NAME):$(TAG) $(REGISTRY)/$(IMAGE_NAME):$(TAG)
	$(CONTAINER_RUNTIME) push $(REGISTRY)/$(IMAGE_NAME):$(TAG)

# Login to ACR
acr-login:
	@echo "Checking registry login status..."
	@if ! $(CONTAINER_RUNTIME) login $(REGISTRY_URL) >/dev/null 2>&1; then \
		echo "Not logged into registry. Please login first."; \
		exit 1; \
	fi	

# Build image directly in ACR
acr-build:
	az acr build --registry $(ACR_NAME) --image $(IMAGE_NAME):$(TAG) .

# Build and push to ACR in one step
acr-push: acr-login push
	@echo "Image $(IMAGE_NAME):$(TAG) built and pushed to $(ACR_NAME).azurecr.io"

# Run container locally (requires GPU support)
run:
	$(CONTAINER_RUNTIME) run --gpus all -p $(PORT):$(PORT) \
		-e SERVER_PORT=$(PORT) \
		-e MODEL_NAME=$(MODEL_NAME) \
		-e SERVER_LOG_LEVEL=info \
		--name $(IMAGE_NAME)-container \
		$(IMAGE_NAME):$(TAG)

# Run development server
dev:
	python -m uvicorn src.main:app --reload --host 0.0.0.0 --port $(PORT)

# Setup development environment
setup-dev:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

# Show help
help:
	@echo "Available targets:"
	@echo "  all            : Build GPU Docker image (default)"
	@echo "  build          : Build GPU Docker image"
	@echo "  push           : Push GPU image to registry"
	@echo "  acr-login      : Login to Azure Container Registry"
	@echo "  acr-build      : Build image directly in Azure Container Registry"
	@echo "  acr-push       : Build and push to Azure Container Registry in one step"
	@echo "  run            : Run container locally (requires GPU support)"
	@echo "  dev            : Run development server"
	@echo "  setup-dev      : Setup development environment"
	@echo "  clean          : Clean up Python artifacts"
	@echo "  test           : Run tests"
	@echo "  help           : Show this help message" 
