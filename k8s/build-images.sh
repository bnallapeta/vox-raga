#!/bin/bash
set -e

# Local script to deploy the image-builder pod and execute build process
# Run this script on your ARM64 Mac to build x86_64 images in a Kubernetes pod

# Variables - adjust these as needed
REGISTRY=${REGISTRY:-"docker.io/yourusername"}
TAG=${TAG:-"0.0.1"}
IMAGE_NAME=${IMAGE_NAME:-"vox-raga"}
CURRENT_DIR=$(pwd)

echo "=== Starting vox-raga build process ==="
echo "Current directory: $CURRENT_DIR"
echo "Registry: $REGISTRY"
echo "Tag: $TAG"

# Step 1: Create the image-builder pod
echo "Creating image-builder pod..."
kubectl apply -f k8s/image-builder-pod.yaml

# Step 2: Wait for pod to be ready
echo "Waiting for image-builder pod to be ready..."
kubectl wait --for=condition=Ready pod/image-builder --timeout=120s

# Step 3: Copy project files to the pod
echo "Copying project files to pod..."
tar -czf /tmp/vox-raga-source.tar.gz --exclude=".git" .
kubectl cp /tmp/vox-raga-source.tar.gz image-builder:/build/
kubectl exec image-builder -- bash -c "mkdir -p /build/vox-raga && tar -xzf /build/vox-raga-source.tar.gz -C /build/vox-raga"

# Step 4: Make the build script executable and copy it
echo "Copying build script to pod..."
chmod +x k8s/image-builder-script.sh
kubectl cp k8s/image-builder-script.sh image-builder:/build/

# Step 5: Execute the build script in the pod
echo "Running build script in pod..."
kubectl exec image-builder -- bash -c "chmod +x /build/image-builder-script.sh && IMAGE_NAME=$IMAGE_NAME TAG=$TAG REGISTRY=$REGISTRY BUILD_DIR=/build /build/image-builder-script.sh"

# Step 6: If a registry was specified, the images should now be pushed
if [ -n "$REGISTRY" ]; then
  echo "Images built and pushed to $REGISTRY/$IMAGE_NAME:$TAG-cpu and $REGISTRY/$IMAGE_NAME:$TAG-gpu"
  echo "You can now use these images in your Kubernetes deployments."
else
  echo "Images built but not pushed (no registry specified)."
fi

# Step 7: Ask if the pod should be deleted
read -p "Do you want to delete the image-builder pod? (y/n): " DELETE_POD
if [ "$DELETE_POD" = "y" ] || [ "$DELETE_POD" = "Y" ]; then
  echo "Deleting image-builder pod..."
  kubectl delete pod image-builder
  echo "Pod deleted."
else
  echo "Pod is still running. You can delete it later with: kubectl delete pod image-builder"
fi

echo "=== Build process complete ===" 