# Building vox-raga Images Across Architectures

This guide provides instructions for building vox-raga Docker images on ARM64 Mac for deployment on x86_64/amd64 platforms.

## The Architecture Challenge

The Coqui TTS base images (`ghcr.io/coqui-ai/tts:v0.22.0` and `ghcr.io/coqui-ai/tts-cpu:v0.22.0`) are only available for x86_64/amd64 architecture. When trying to build on an ARM64 Mac, you'll encounter architecture compatibility issues.

This repository provides two approaches to solve this problem:

1. **Kubernetes Pod Builder**: Build images in an x86_64 Kubernetes pod
2. **Podman Container Builder**: Build images in a Podman container

## Option 1: Building in a Kubernetes Pod

This approach requires:
- Access to a Kubernetes cluster with x86_64/amd64 nodes
- `kubectl` configured to connect to your cluster

### Steps:

1. Make sure your kubectl context is set to the correct cluster:
   ```bash
   kubectl config current-context
   ```

2. Run the build script:
   ```bash
   ./k8s/build-images.sh
   ```

3. The script will:
   - Create a pod called `image-builder` in your cluster
   - Copy your project files to the pod
   - Install Docker and other dependencies
   - Build the CPU and GPU images
   - Push the images to your registry (if specified)
   - Offer to delete the pod when complete

### Customizing the Build:

You can customize the build by setting environment variables:

```bash
REGISTRY=docker.io/yourusername TAG=0.0.2 IMAGE_NAME=vox-raga ./k8s/build-images.sh
```

## Option 2: Building in a Podman Container

This approach uses Podman directly on your Mac:

1. Make sure Podman is installed and running:
   ```bash
   podman info
   ```

2. Run the Podman build script:
   ```bash
   ./k8s/podman-build-images.sh
   ```

3. The script will:
   - Create a Podman container
   - Copy your project files to the container
   - Install Docker and other dependencies
   - Build the CPU and GPU images
   - Push the images to your registry (if specified)
   - Offer to delete the container when complete

### Customizing the Build:

You can customize the build by setting environment variables:

```bash
REGISTRY=docker.io/yourusername TAG=0.0.2 IMAGE_NAME=vox-raga ./k8s/podman-build-images.sh
```

## Using Your Built Images

After building and pushing to a registry, update your Kubernetes deployment files:

```yaml
# In k8s/vox-raga-deployment.yaml and k8s/vox-raga-cpu-deployment.yaml
containers:
- name: vox-raga
  image: docker.io/yourusername/vox-raga:0.0.1-gpu  # or -cpu for CPU version
```

## Troubleshooting

### Common Issues:

1. **"no image found in image index for architecture"**:
   - This is the original issue these scripts solve - the Coqui TTS base images are x86_64 only.

2. **Docker-in-Docker issues**:
   - If you see Docker connectivity problems, make sure the container has proper privileges.

3. **Push authentication failures**:
   - Set `REGISTRY_USER` and `REGISTRY_PASS` environment variables for authentication.

4. **Pod/container creation failures**:
   - Check that your Kubernetes cluster has x86_64/amd64 nodes available.
   - Ensure you have permissions to create privileged containers if necessary.

## Build Once, Deploy Many Times

The recommended workflow is:

1. Build images once and push to a registry
2. Use those images in your Kubernetes deployments
3. Only rebuild when you make changes to your code 