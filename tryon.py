import os
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

def run_virtual_tryon(person_bytes: bytes, person_name: str,
                      garment_bytes: bytes, garment_name: str) -> Image.Image:
    """
    Calls a compatible virtual try-on inference endpoint.

    Expected request fields by default:
      person_image: uploaded person image
      garment_image: uploaded clothing image
    Expected response: image bytes (PNG/JPEG) or JSON containing an image URL is NOT
    automatically supported; adapt this function to the provider's API contract.
    """
    endpoint = os.getenv("TRYON_API_URL", "").strip()
    if not endpoint:
        raise RuntimeError("TRYON_API_URL is missing in .env")
    headers = {}
    api_key = os.getenv("TRYON_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    files = {
        "person_image": (person_name, BytesIO(person_bytes), "application/octet-stream"),
        "garment_image": (garment_name, BytesIO(garment_bytes), "application/octet-stream"),
    }
    response = requests.post(endpoint, headers=headers, files=files, timeout=180)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "image" not in content_type:
        raise RuntimeError(
            f"Endpoint returned '{content_type or 'unknown content type'}', not an image. "
            "Update tryon.py to match your provider's response format."
        )
    try:
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        raise RuntimeError("Could not decode the endpoint response as an image.") from exc
