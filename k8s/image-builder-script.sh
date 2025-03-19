#!/bin/bash
set -e

# Image Builder Script for vox-raga
# This script runs inside the image-builder pod and builds Docker images for x86_64 architecture

# Variables
IMAGE_NAME=${IMAGE_NAME:-vox-raga}
TAG=${TAG:-0.0.1}
BUILD_DIR=${BUILD_DIR:-/build}

# Setup function for installing dependencies
setup_environment() {
  echo "Setting up build environment..."
  apt-get update
  apt-get install -y apt-transport-https ca-certificates curl software-properties-common git make zip unzip
  
  # Install Docker
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
  add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io
  
  # Check if Docker is working
  docker info || { echo "Docker is not running properly"; exit 1; }
  
  echo "Environment setup completed."
}

# Clone repository
clone_repo() {
  echo "Cloning vox-raga repository..."
  cd ${BUILD_DIR}
  if [ -d "vox-raga" ]; then
    echo "Repository already exists, updating..."
    cd vox-raga
    git pull
  else
    echo "Fresh clone of repository..."
    git clone https://github.com/yourusername/vox-raga.git
    cd vox-raga
  fi
  
  echo "Repository ready at ${BUILD_DIR}/vox-raga"
}

# Build images
build_images() {
  echo "Building Docker images..."
  cd ${BUILD_DIR}/vox-raga
  
  echo "Building CPU image..."
  docker build -t ${IMAGE_NAME}:${TAG}-cpu -f Dockerfile.cpu .
  
  echo "Building GPU image..."
  docker build -t ${IMAGE_NAME}:${TAG}-gpu -f Dockerfile.gpu .
  
  echo "Images built successfully."
}

# Push images to registry
push_images() {
  if [ -z "$REGISTRY" ]; then
    echo "No registry specified, skipping push."
    return
  fi
  
  echo "Pushing images to registry: $REGISTRY"
  
  docker tag ${IMAGE_NAME}:${TAG}-cpu ${REGISTRY}/${IMAGE_NAME}:${TAG}-cpu
  docker tag ${IMAGE_NAME}:${TAG}-gpu ${REGISTRY}/${IMAGE_NAME}:${TAG}-gpu
  
  # Optional: Login to registry
  if [ -n "$REGISTRY_USER" ] && [ -n "$REGISTRY_PASS" ]; then
    echo "Logging in to registry..."
    echo "$REGISTRY_PASS" | docker login ${REGISTRY} -u ${REGISTRY_USER} --password-stdin
  fi
  
  docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}-cpu
  docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}-gpu
  
  echo "Images pushed successfully."
}

# Main execution
main() {
  echo "Starting build process..."
  setup_environment
  clone_repo
  build_images
  
  # Push if registry is specified
  if [ -n "$REGISTRY" ]; then
    push_images
  fi
  
  echo "Build process completed successfully."
}

# Run main function
main 