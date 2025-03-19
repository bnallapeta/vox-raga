#!/usr/bin/env python3
"""
Helper script to download and set up TTS models for local development.
"""
import os
import json
import shutil
from TTS.utils.manage import ModelManager

def setup_model(model_name="tts_models/en/vctk/vits", output_dir="/tmp/tts_models"):
    """Download and set up a TTS model for local development."""
    print(f"Setting up model {model_name} in {output_dir}")
    
    # Create model directory structure
    model_parts = model_name.split("/")
    if len(model_parts) < 4:
        raise ValueError(f"Invalid model name: {model_name}")
    
    model_type, lang, dataset, model = model_parts
    model_dir = os.path.join(output_dir, model_type, lang, dataset, model)
    os.makedirs(model_dir, exist_ok=True)
    
    # Download model
    print("Downloading model...")
    model_manager = ModelManager()
    model_path, config_path, model_item = model_manager.download_model(model_name)
    print(f"Model downloaded to {model_path}")
    print(f"Config downloaded to {config_path}")
    
    # Copy model files
    model_filename = os.path.basename(model_path)
    config_filename = os.path.basename(config_path)
    
    target_model_path = os.path.join(model_dir, "model_file.pth")
    target_config_path = os.path.join(model_dir, "config.json")
    
    shutil.copy(model_path, target_model_path)
    shutil.copy(config_path, target_config_path)
    print(f"Copied model file to {target_model_path}")
    print(f"Copied config file to {target_config_path}")
    
    # Create models.json file
    models_json = {
        "tts_models": {
            lang: {
                dataset: {
                    model: {
                        "description": f"TTS model for {lang} using {model}",
                        "model_name": model_name,
                        "default_vocoder": None
                    }
                }
            }
        }
    }
    
    with open(os.path.join(output_dir, "models.json"), "w") as f:
        json.dump(models_json, f, indent=4)
    print(f"Created models.json file at {os.path.join(output_dir, 'models.json')}")
    
    # List the directory structure
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")
    
    print("\nModel setup completed successfully!")

if __name__ == "__main__":
    setup_model() 