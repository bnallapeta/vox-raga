# Phony targets declaration
.PHONY: all build build-cpu build-gpu push push-cpu push-gpu run clean help \
        build-amd64-cpu build-amd64-gpu

# Variables
IMAGE_NAME ?= vox-raga
TAG ?= 0.0.1
PORT ?= 8888
MODEL_NAME ?= tts_models/en/vctk/vits

# Container runtime configuration
CONTAINER_RUNTIME ?= $(shell which podman 2>/dev/null || which docker 2>/dev/null)
REGISTRY ?= bnracr.azurecr.io

# Architecture detection
ARCH := $(shell uname -m)

# Default target
all: build-cpu

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

# Simple build - both CPU and GPU
build: build-cpu build-gpu

# Build CPU version (for native architecture)
build-cpu:
	$(CONTAINER_RUNTIME) build -t $(IMAGE_NAME):$(TAG)-cpu -f Dockerfile.cpu .

# Build GPU version (for native architecture)
build-gpu:
	$(CONTAINER_RUNTIME) build -t $(IMAGE_NAME):$(TAG)-gpu -f Dockerfile.gpu .

# Build CPU version specifically for AMD64 architecture
# This is needed when building on Mac ARM64 for x86_64 targets
build-amd64-cpu:
ifeq ($(ARCH),arm64)
	@echo "Building CPU image for AMD64 architecture from ARM64 Mac..."
	# For Docker, use buildx
	if [ "$(CONTAINER_RUNTIME)" = "docker" ]; then \
		docker buildx build --platform=linux/amd64 -t $(IMAGE_NAME):$(TAG)-cpu -f Dockerfile.cpu --load .; \
	else \
		# For Podman, just build and warn
		echo "WARNING: Building with Podman on ARM64 Mac. The resulting image may not work on AMD64 servers."; \
		podman build -t $(IMAGE_NAME):$(TAG)-cpu -f Dockerfile.cpu .; \
	fi
else
	# On AMD64, just do a normal build
	$(CONTAINER_RUNTIME) build -t $(IMAGE_NAME):$(TAG)-cpu -f Dockerfile.cpu .
endif

# Build GPU version specifically for AMD64 architecture
# This is needed when building on Mac ARM64 for x86_64 targets
build-amd64-gpu:
ifeq ($(ARCH),arm64)
	@echo "Building GPU image for AMD64 architecture from ARM64 Mac..."
	# For Docker, use buildx
	if [ "$(CONTAINER_RUNTIME)" = "docker" ]; then \
		docker buildx build --platform=linux/amd64 -t $(IMAGE_NAME):$(TAG)-gpu -f Dockerfile.gpu --load .; \
	else \
		# For Podman, just build and warn
		echo "WARNING: Building with Podman on ARM64 Mac. The resulting image may not work on AMD64 servers."; \
		podman build -t $(IMAGE_NAME):$(TAG)-gpu -f Dockerfile.gpu .; \
	fi
else
	# On AMD64, just do a normal build
	$(CONTAINER_RUNTIME) build -t $(IMAGE_NAME):$(TAG)-gpu -f Dockerfile.gpu .
endif

# Push both images
push: push-cpu push-gpu

# Push CPU image
push-cpu:
	$(CONTAINER_RUNTIME) tag $(IMAGE_NAME):$(TAG)-cpu $(REGISTRY)/$(IMAGE_NAME):$(TAG)-cpu
	$(CONTAINER_RUNTIME) push $(REGISTRY)/$(IMAGE_NAME):$(TAG)-cpu

# Push GPU image
push-gpu:
	$(CONTAINER_RUNTIME) tag $(IMAGE_NAME):$(TAG)-gpu $(REGISTRY)/$(IMAGE_NAME):$(TAG)-gpu
	$(CONTAINER_RUNTIME) push $(REGISTRY)/$(IMAGE_NAME):$(TAG)-gpu

# Run container locally
run:
	$(CONTAINER_RUNTIME) run -p $(PORT):$(PORT) \
		-e SERVER_PORT=$(PORT) \
		-e MODEL_NAME=$(MODEL_NAME) \
		-e SERVER_LOG_LEVEL=info \
		--name $(IMAGE_NAME)-container \
		$(IMAGE_NAME):$(TAG)-cpu

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
	@echo "  all            : Build CPU Docker image (default)"
	@echo "  build          : Build both CPU and GPU images"
	@echo "  build-cpu      : Build CPU Docker image"
	@echo "  build-gpu      : Build GPU Docker image" 
	@echo "  build-amd64-cpu: Build CPU image specifically for AMD64 (for ARM64 Mac)"
	@echo "  build-amd64-gpu: Build GPU image specifically for AMD64 (for ARM64 Mac)"
	@echo "  push           : Push both CPU and GPU images to registry"
	@echo "  push-cpu       : Push CPU image to registry"
	@echo "  push-gpu       : Push GPU image to registry"
	@echo "  run            : Run container locally (CPU version)"
	@echo "  dev            : Run development server"
	@echo "  setup-dev      : Setup development environment"
	@echo "  clean          : Clean up Python artifacts"
	@echo "  test           : Run tests"
	@echo "  help           : Show this help message" 
