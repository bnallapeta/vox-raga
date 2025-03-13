from TTS.utils.manage import ModelManager
import json

mm = ModelManager()
with open('/tmp/tts_models/models.json', 'w') as f:
    json.dump(mm.models_dict, f, indent=2)

print("Models.json file created successfully at /tmp/tts_models/models.json") 