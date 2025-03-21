# Vox-Raga with XTTS-v2 Model

This README provides instructions for setting up and running the Vox-Raga TTS service with the XTTS-v2 model using volume mounting in Kubernetes.

## Model Setup

### 1. Download the XTTS-v2 model files

You need to download these model files from Hugging Face:

1. Create a Hugging Face account if you don't have one
2. Visit https://huggingface.co/coqui/XTTS-v2
3. Download the following files:
   - `config.json` (4.37 KB)
   - `model.pth` (1.87 GB)
   - `speakers_xtts.pth` (7.75 MB)
   - `dvae.pth` (211 MB)
   - `vocab.json` (361 KB)

### 2. Place the files in the correct directory structure

Create the following directory structure in your project folder:

```
models/
└── tts_models/
    └── multilingual/
        └── multi-dataset/
            └── xtts_v2/
                ├── config.json
                ├── model.pth
                ├── speakers_xtts.pth
                ├── dvae.pth
                └── vocab.json
```

### 3. Create a models.json file

Create a file at `models/models.json` with the following content:

```json
{
  "tts_models": {
    "multilingual": {
      "multi-dataset": {
        "xtts_v2": {
          "description": "XTTS v2 multilingual TTS model",
          "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
          "default_vocoder": null
        }
      }
    }
  }
}
```

## Building and Running the Container

### 1. Build the Docker image

```bash
docker build -f Dockerfile.gpu -t bnracr.azurecr.io/vox-raga:0.0.8 .
```

### 2. Push the image to your registry

```bash
docker push bnracr.azurecr.io/vox-raga:0.0.8
```

### 3. Deploy to Kubernetes

```bash
kubectl apply -f k8s/vox-raga-gpu-reduced.yaml
```

## How It Works

- The Kubernetes deployment mounts the local `models` directory from your host (`/home/bnallapeta/work/vox-raga/models`) into the container at `/app/models`
- The container is configured to use the XTTS-v2 model with the environment variable `MODEL_NAME=tts_models/multilingual/multi-dataset/xtts_v2`
- The TTS service looks for the model files in the mounted volume and uses them for speech synthesis

## Advantages of this Approach

1. **Separation of concerns**: Model files are kept separate from application code
2. **No download at runtime**: No need to download large model files at container startup
3. **Version control friendly**: You can version control the model files separately
4. **Reusability**: The same model files can be used by multiple containers
5. **Faster startup**: Container starts immediately without downloading models

## Troubleshooting

If you encounter issues with the TTS service:

- Check that all model files are correctly placed in the mounted volume
- Verify that the mounted directory has proper permissions
- Check the container logs for any errors: `kubectl logs deployment/vox-raga-gpu`
- Ensure your Kubernetes node has enough GPU memory for the XTTS-v2 model (it's larger than VITS) 