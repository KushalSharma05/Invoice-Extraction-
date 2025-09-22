import fitz  # PyMuPDF
import cv2
import numpy as np
import json
import requests
import base64
import tempfile
import os
import logging

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Functions
# -----------------------------

def sharpen_image(image):
    """Sharpen an OpenCV image."""
    logging.info('Sharpening image')
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def pix_to_cv2(pix):
    """Convert PyMuPDF pixmap to OpenCV BGR image."""
    logging.info('Converting pixmap to OpenCV image')
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:  # BGRA to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif pix.n == 1:  # Grayscale to BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def run_ollama_model(image_path):
    """Send image to Ollama API and return JSON string."""
    logging.info(f'Running Ollama model on image: {image_path}')
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')

    prompt = (
        "Extract the following fields from this image as JSON. "
        "If a field is missing, return an empty string. "
        "Fields: supplier_name, gateentry_no, po_no, vehicle_no, po_date, invoice_no, "
        "invoice_date, challan_no, challan_date, material_name, material_unit, challan_qty, "
        "material_rate, gst_no, ewaybill_no, ewaybill_date, lr_no, lr_date, plant_no."
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5vl:7b",
                "prompt": prompt,
                "images": [base64_image],
                "stream": False
            }
        )
        response.raise_for_status()
        data = response.json()
        if 'response' not in data:
            logging.warning(f"No 'response' key in API output: {data}")
            return "{}"
        logging.info("Model call successful")
        return data['response']
    except Exception as e:
        logging.error(f"Ollama API call failed: {e}")
        return "{}"

# -----------------------------
# Main PDF Processing
# -----------------------------

def process_pdf(pdf_path, output_json_path='merged_results.json'):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logging.info(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    merged_results = []

    for page_num in range(len(doc)):
        logging.info(f"Processing page {page_num + 1}/{len(doc)}")
        page = doc.load_page(page_num)

        # Render page at 400 dpi
        zoom = 400 / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert and sharpen
        cv_img = pix_to_cv2(pix)
        sharp_img = sharpen_image(cv_img)

        # Save temp image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            temp_path = tmpfile.name
            cv2.imwrite(temp_path, sharp_img)
            logging.info(f"Saved temporary image: {temp_path}")

        # Run model
        json_str = run_ollama_model(temp_path)
        logging.info(f"Raw model output: {json_str}")

        # Parse JSON safely
        try:
            json_data = json.loads(json_str)
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON for page {page_num + 1}")
            json_data = {}

        merged_results.append(json_data)

        # Remove temp image
        os.remove(temp_path)
        logging.info(f"Removed temporary image: {temp_path}")

    # Merge JSONs into one clean object
    fields = [
        "supplier_name", "gateentry_no", "po_no", "vehicle_no", "po_date", "invoice_no",
        "invoice_date", "challan_no", "challan_date", "material_name", "material_unit",
        "challan_qty", "material_rate", "gst_no", "ewaybill_no", "ewaybill_date",
        "lr_no", "lr_date", "plant_no"
    ]

    # Remove duplicates in fields
    unique_fields = list(dict.fromkeys(fields))
    merged_json = {}

    logging.info("Merging results from all pages")
    for field in unique_fields:
        values = [res.get(field, "") for res in merged_results if res.get(field)]
        if len(values) == 1:
            merged_json[field] = values[0]
        elif len(values) > 1:
            merged_json[field] = values
        else:
            merged_json[field] = ""

    # Save final JSON
    with open(output_json_path, "w") as f:
        json.dump(merged_json, f, indent=4)
    logging.info(f"Merged JSON saved: {output_json_path}")

    return merged_json

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    pdf_path = "1054_20250913_ea51.pdf"  # Replace with your PDF
    result = process_pdf(pdf_path)
    print(json.dumps(result, indent=4))
