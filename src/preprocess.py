import re
import unicodedata

def normalize_vietnamese_text(text):
    """Normalize Vietnamese text for better processing."""
    # Normalize Unicode
    text = unicodedata.normalize('NFC', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_json_data(json_data):
    """Preprocess all text fields in the JSON data."""
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if isinstance(value, str):
                json_data[key] = normalize_vietnamese_text(value)
            else:
                json_data[key] = preprocess_json_data(value)
    elif isinstance(json_data, list):
        for i, item in enumerate(json_data):
            json_data[i] = preprocess_json_data(item)
    
    return json_data
