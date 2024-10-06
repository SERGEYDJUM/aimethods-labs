import requests
from utils import *
from cv2 import Mat
from datetime import datetime

def api_l(file_path: str) -> Mat:
    url = "https://ai-image-upscaler1.p.rapidapi.com/v1"

    files = {
        'image': ("file.png", open(file_path, 'rb'), "image/png"),
    }
    
    data = {
        "scale_factor": "2"
    }
    
    headers = {
        "x-rapidapi-key": "fd9c4fcbcamsh878ca3f7f0c633fp14bc03jsnb84bc4e6dfa1",
	    "x-rapidapi-host": "ai-image-upscaler1.p.rapidapi.com",
    }

    # 3-6
    start = datetime.now()
    response = requests.post(url, data=data, files=files, headers=headers)
    dur = datetime.now() - start
    print(f"API L: Got response after {dur.total_seconds()}")
    
    if response.status_code != 200:
        raise RuntimeError(f"request failed with code: {response.status_code}")
    
    return b64_to_cv(response.json()["result_base64"])


def api_r(file_path: str, strength=0.3) -> Mat:
    url = "https://vision-ai-api.p.rapidapi.com/imgupscalerform"

    files = {
        'file': ("file.png", open(file_path, 'rb'), "image/png"),
    }
    
    data = {
        "resolution": "512",
        "strength": f"{strength:.2f}",
        "hdr_effect": "0"
    }
    
    headers = {
        "x-rapidapi-key": "fd9c4fcbcamsh878ca3f7f0c633fp14bc03jsnb84bc4e6dfa1",
	    "x-rapidapi-host": "vision-ai-api.p.rapidapi.com",
    }

    # 18-40
    start = datetime.now()
    response = requests.post(url, data=data, files=files, headers=headers)
    dur = datetime.now() - start
    print(f"API R: Got response after {dur.total_seconds()}")
    
    if response.status_code != 200:
        raise RuntimeError(f"request failed with code: {response.status_code}")
    
    response = requests.get(response.json()["generated_image"], stream=True)
    
    if response.status_code != 200:
        raise RuntimeError(f"request failed with code: {response.status_code}")
    
    return retrieve_image(response.raw)