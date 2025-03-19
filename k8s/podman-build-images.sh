#!/bin/bash
set -e

# Local script to build images using a Podman container
# Run this script on your ARM64 Mac to build x86_64 images in a Podman container

# Variables - adjust these as needed
REGISTRY=${REGISTRY:-"docker.io/yourusername"}
TAG=${TAG:-"0.0.1"}
IMAGE_NAME=${IMAGE_NAME:-"vox-raga"}
CONTAINER_NAME="vox-raga-builder"
CURRENT_DIR=$(pwd)

echo "=== Starting vox-raga build process with Podman ==="
echo "Current directory: $CURRENT_DIR"
echo "Registry: $REGISTRY"
echo "Tag: $TAG"

# Step 1: Check if the builder container already exists
if podman container exists $CONTAINER_NAME; then
  echo "Builder container already exists, removing it..."
  podman rm -f $CONTAINER_NAME
fi

# Step 2: Create the builder container
echo "Creating builder container..."
podman run -d --name $CONTAINER_NAME \
  --privileged \
  -v /var/run/podman/podman.sock:/var/run/docker.sock \
  ubuntu:22.04 sleep 3600

# Step 3: Wait a moment for container to start
echo "Waiting for container to start..."
sleep 5

# Step 4: Copy project files to the container
echo "Copying project files to container..."
tar -czf /tmp/vox-raga-source.tar.gz --exclude=".git" .
podman cp /tmp/vox-raga-source.tar.gz $CONTAINER_NAME:/tmp/
podman exec $CONTAINER_NAME bash -c "mkdir -p /build/vox-raga && tar -xzf /tmp/vox-raga-source.tar.gz -C /build/vox-raga"

# Step 5: Create the build script directly in the container
echo "Creating build script in container..."
cat > /tmp/image-builder-script.sh << 'EOF'
#!/bin/bash
set -e

# Image Builder Script for vox-raga
# This script runs inside a container and builds Docker images for x86_64 architecture

# Variables
IMAGE_NAME=${IMAGE_NAME:-vox-raga}
TAG=${TAG:-0.0.1}
BUILD_DIR=${BUILD_DIR:-/build}

# Setup function for installing dependencies
setup_environment() {
  echo "Setting up build environment..."
  apt-get update
  apt-get install -y apt-transport-https ca-certificates curl software-properties-common git make zip unzip lsb-release
  
  # Install Docker
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
  add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io
  
  # Check if Docker is working
  docker info || { echo "Docker is not running properly"; exit 1; }
  
  echo "Environment setup completed."
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
  build_images
  
  # Push if registry is specified
  if [ -n "$REGISTRY" ]; then
    push_images
  fi
  
  echo "Build process completed successfully."
}

# Run main function
main
EOF

# Step 6: Copy build script to container and make it executable
podman cp /tmp/image-builder-script.sh $CONTAINER_NAME:/build/
podman exec $CONTAINER_NAME chmod +x /build/image-builder-script.sh

# Step 7: Execute the build script in the container
echo "Running build script in container..."
podman exec -e IMAGE_NAME=$IMAGE_NAME -e TAG=$TAG -e REGISTRY=$REGISTRY -e BUILD_DIR=/build $CONTAINER_NAME /build/image-builder-script.sh

# Step 8: If a registry was specified, the images should now be pushed
if [ -n "$REGISTRY" ]; then
  echo "Images built and pushed to $REGISTRY/$IMAGE_NAME:$TAG-cpu and $REGISTRY/$IMAGE_NAME:$TAG-gpu"
  echo "You can now use these images in your Kubernetes deployments."
else
  echo "Images built but not pushed (no registry specified)."
fi

# Step 9: Ask if the container should be deleted
read -p "Do you want to delete the builder container? (y/n): " DELETE_CONTAINER
if [ "$DELETE_CONTAINER" = "y" ] || [ "$DELETE_CONTAINER" = "Y" ]; then
  echo "Deleting builder container..."
  podman rm -f $CONTAINER_NAME
  echo "Container deleted."
else
  echo "Container is still running. You can delete it later with: podman rm -f $CONTAINER_NAME"
fi

echo "=== Build process complete ===" 