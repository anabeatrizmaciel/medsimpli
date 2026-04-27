import os
import json

DATA_INPUT_DIR = "data/processed/json"
DATA_OUTPUT_DIR = "data/cleaned"

os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

if not os.path.exists(DATA_INPUT_DIR):
    raise FileNotFoundError(f"Diretório de entrada '{DATA_INPUT_DIR}' não encontrado.")
else:
    for filename in os.listdir(DATA_INPUT_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(DATA_INPUT_DIR, filename), "r", encoding="utf-8") as f:
                file_data = json.load(f)
                title = file_data["title"]
                text = file_data["text"]
                clean_text = text.replace("\n", " ").replace(" ,", ",").replace(" .", ".").strip()
                output_data = {"title": title, "text": clean_text}
                with open(os.path.join(DATA_OUTPUT_DIR, filename), "w", encoding="utf-8") as out_f:
                    json.dump(output_data, out_f, ensure_ascii=False, indent=4)