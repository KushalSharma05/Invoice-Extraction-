import cv2
import json
import ollama
import re
from typing import Optional, Dict, Any
from PIL import Image
import os
import logging


# -----------------------------
# LOGGING CONFIG
# -----------------------------
logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for very detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# -----------------------------
# Robust JSON Extractor
# -----------------------------
def _robust_json_extractor(text: str) -> Optional[Dict[str, Any]]:
    """
    More robustly extracts a JSON object from a string, handling markdown fences and extra text.
    """
    # Remove ```json fences
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)

    # Find JSON object in text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.error("❌ JSON EXTRACTION FAILED: No JSON object found in the text.")
        return None

    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"⚠️ JSON DECODE ERROR: {e}. Problematic string: {json_str}")
        return None


# -----------------------------
# STEP 1: Preprocess Image
# -----------------------------
def preprocess_image(input_path: str, output_path: str, dpi: int = 600, scale: int = 3):
    """
    Preprocess the image for better OCR/vision model accuracy.
    - Converts to grayscale
    - Sharpens text
    - Upscales for virtual high DPI
    - Saves with embedded DPI metadata
    """
    logger.info(f"Reading input image: {input_path}")
    image = cv2.imread(input_path)
    if image is None:
        logger.error(f"❌ Could not read image: {input_path}")
        raise FileNotFoundError(f"Could not read image: {input_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply unsharp masking
    gaussian = cv2.GaussianBlur(gray, (9, 9), 10)
    sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)

    # Upscale for virtual high DPI
    width = int(sharpened.shape[1] * scale)
    height = int(sharpened.shape[0] * scale)
    high_res = cv2.resize(sharpened, (width, height), interpolation=cv2.INTER_CUBIC)

    temp_path = output_path.replace(".png", "_temp.png")
    cv2.imwrite(temp_path, high_res)

    # Save with DPI metadata
    img = Image.open(temp_path)
    img.save(output_path, dpi=(dpi, dpi))

    if os.path.exists(temp_path):
        os.remove(temp_path)

    logger.info(f"✅ Processed image saved at {output_path} with {dpi} DPI")


# -----------------------------
# STEP 2: Query Llama Vision (One Go)
# -----------------------------
def extract_fields(image_path: str) -> dict:
    """
    Extracts all required fields in one go from the invoice using Llama 3.2 Vision.
    Returns a dictionary.
    """
    prompt = """
    You are an OCR and data extraction assistant.
    Extract the following fields from the invoice and return ONLY a valid JSON object:

    {
        "invoice_no": "",
        "ewaybill_no": "",
        "po_no": "",
        "gst_no": "",
        "quantity": "",
        "supplier_name": "",
        "consignee_details": "",
        "supplier_address": ""
    }
    """

    logger.info("📤 Sending single extraction request to Llama Vision")
    response = ollama.chat(
        model="llama3.2-vision:latest",
        messages=[
            {"role": "user", "content": prompt, "images": [image_path]}
        ]
    )

    raw_output = response["message"]["content"]
    logger.debug(f"Raw model output: {raw_output}")

    extracted = _robust_json_extractor(raw_output)
    if extracted:
        logger.info("✅ Successfully extracted JSON fields")
        return extracted
    else:
        return {"error": "Invalid JSON returned by model", "raw_output": raw_output}


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    input_img = r"D:\C Drive Data_27-06-2025\Desktop\abc1.png"
    output_img = r"D:\C Drive Data_27-06-2025\Desktop\abc1_processed.png"

    try:
        # Preprocess
        preprocess_image(input_img, output_img, dpi=600, scale=3)

        # Extract fields in one go
        results = extract_fields(output_img)

        # Save results
        with open("extracted_fields.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        logger.info("✅ Extraction completed. Results saved to extracted_fields.json")

    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")
