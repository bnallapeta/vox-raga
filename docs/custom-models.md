# Using Custom TTS Models

This guide explains how to use custom TTS models with the TTS service.

## Overview

The TTS service supports both:
- **Built-in models**: Pre-packaged models included in the container image
- **Custom models**: User-provided models mounted through Kubernetes volumes

## Custom Model Directory Structure

Custom models should follow this directory structure:

```
/your-models-directory/
├── my_custom_models/
│   ├── en/
│   │   ├── my_model/
│   │   │   ├── vits/
│   │   │   │   ├── config.json  # Model configuration
│   │   │   │   ├── model.pth    # Model weights
│   │   │   │   └── speakers.json  # (optional) Speaker metadata
│   │   │   └── ... (other model versions)
│   │   └── ... (other datasets)
│   └── ... (other languages)
└── models.json  # Model registry file
```

The `models.json` file should follow this format:

```json
{
  "my_custom_models": {
    "en": {
      "my_model": {
        "vits": {
          "description": "My custom VITS model",
          "model_name": "my_custom_models/en/my_model/vits",
          "default_vocoder": null
        }
      }
    }
  }
}
```

## Using Custom Models in Kubernetes

To use custom models, you need to:

1. Create a volume containing your custom models
2. Configure the TTS service to use this volume

### Example: Using a PersistentVolumeClaim

1. Create a PVC for your models:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: tts-models-pvc
  namespace: default
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 1Gi
  storageClassName: standard
```

2. Populate the PVC with your models (using a Job, InitContainer, or manual methods)

3. Deploy the TTS service with custom model configuration:

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "tts-service-custom"
  namespace: "default"
  annotations:
    sidecar.istio.io/inject: "true"
    serving.kserve.io/deploymentMode: "RawDeployment"
spec:
  predictor:
    containers:
      - name: tts-service
        image: bnracr.azurecr.io/tts-service:latest
        env:
        - name: MODEL_NAME
          value: "my_custom_models/en/my_model/vits"
        - name: USE_CUSTOM_MODEL
          value: "true"
        volumeMounts:
        - name: model-cache
          mountPath: "/tmp/tts_models"
        - name: custom-models
          mountPath: "/mnt/models"
          readOnly: true
    volumes:
    - name: model-cache
      emptyDir: {}
    - name: custom-models
      persistentVolumeClaim:
        claimName: tts-models-pvc
```

## Converting Existing Models

To convert an existing TTS model for use with this service:

1. Ensure you have the model weights (`.pth` file) and configuration (`.json` file)
2. Organize them in the directory structure shown above
3. Create a `models.json` file referencing your model
4. Upload all files to a volume accessible by Kubernetes

## Testing Your Custom Model

1. Deploy the TTS service with your custom model configuration
2. Once the service is running, test it with:

```bash
curl -X POST "http://<service-url>/api/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a custom model test", "options": {"voice": "default"}}'
```

## Troubleshooting

If your custom model is not being loaded correctly:

1. Check that the directory structure matches the expected format
2. Verify that the `models.json` file is formatted correctly
3. Ensure the `MODEL_NAME` environment variable matches the path in your `models.json` file
4. Check the logs for any error messages:
   ```bash
   kubectl logs -f deployment/tts-service
   ``` 