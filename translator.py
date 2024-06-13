import requests
import json
from deep_translator import GoogleTranslator
from tqdm import tqdm

# Function to download the file
def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

# Download the JSON file
url = "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"
local_filename = "ontology.json"
download_file(url, local_filename)

# Read the JSON file
with open(local_filename, 'r', encoding='utf-8') as f:
    ontology = json.load(f)

translator = GoogleTranslator(source='en', target='tr')

# Translation function
def translate_text(text):
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"Error translating text: {text}\n{e}")
        return text

# Translate with progress bar
translated_ontology = []
total_entries = len(ontology)
for entry in tqdm(ontology, desc="Translating", unit="entry"):
    translated_entry = entry.copy()
    if 'name' in entry:
        translated_entry['name'] = translate_text(entry['name'])
    if 'description' in entry:
        translated_entry['description'] = translate_text(entry['description'])
    translated_ontology.append(translated_entry)

# Save the translated JSON file
with open('translated_ontology.json', 'w', encoding='utf-8') as f:
    json.dump(translated_ontology, f, ensure_ascii=False, indent=2)

print("Translation completed and saved to 'translated_ontology.json'.")
