import os
import base64
from flask import Flask, request, jsonify, send_from_directory
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")

VISION_KEY = os.environ.get("VISION_KEY", "")
VISION_ENDPOINT = os.environ.get("VISION_ENDPOINT", "").rstrip("/")

SUPPORTED_ANALYSIS_MODES = {
    "general": ["Tags", "Read"],
    "faces": ["Tags", "Read", "People"],
    "landmarks": ["Tags", "Read"],
    "brands": ["Tags", "Read"],
    "all": ["Tags", "Read", "People"],
}


def normalize_features(raw_features):
    if isinstance(raw_features, str):
        raw_features = [raw_features]

    if not raw_features:
        return SUPPORTED_ANALYSIS_MODES["general"]

    features = []
    for item in raw_features:
        if not isinstance(item, str):
            continue
        feature = item.strip()
        if not feature:
            continue
        key = feature.lower()
        if key in {"general", "default"}:
            features.extend(SUPPORTED_ANALYSIS_MODES["general"])
            continue
        if key in {"face", "faces", "people"}:
            features.append("People")
            continue
        if key in {"landmark", "landmarks"}:
            features.append("Tags")
            continue
        if key in {"brand", "brands"}:
            features.append("Tags")
            continue
        if key in {"caption", "description"}:
            features.append("Caption")
            continue
        if key in {"tag", "tags"}:
            features.append("Tags")
            continue
        if key in {"read", "ocr"}:
            features.append("Read")
            continue
        if key in SUPPORTED_ANALYSIS_MODES:
            features.extend(SUPPORTED_ANALYSIS_MODES[key])
            continue
        features.append(feature)

    if not features:
        return SUPPORTED_ANALYSIS_MODES["general"]

    # deduplicate while preserving order
    deduped = []
    seen = set()
    for feature in features:
        if feature not in seen:
            deduped.append(feature)
            seen.add(feature)
    return deduped


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if not VISION_KEY or not VISION_ENDPOINT:
        return jsonify({
            "error": "Azure Vision not configured. Set VISION_KEY and "
                     "VISION_ENDPOINT in App Service -> Configuration -> "
                     "Application settings, then restart the app."
        }), 500

    data = request.get_json(silent=True) or {}
    image_url = data.get("url")
    image_base64 = data.get("image_base64")
    analysis_mode = data.get("analysis_mode", "general")
    requested_features = normalize_features(data.get("features") or analysis_mode)

    api_url = (
        f"{VISION_ENDPOINT}/computervision/imageanalysis:analyze"
        f"?api-version=2023-10-01&features={','.join(requested_features)}"
    )
    headers = {"Ocp-Apim-Subscription-Key": VISION_KEY}

    try:
        if image_url:
            headers["Content-Type"] = "application/json"
            resp = requests.post(api_url, headers=headers, json={"url": image_url}, timeout=20)
        elif image_base64:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_base64)
            headers["Content-Type"] = "application/octet-stream"
            resp = requests.post(api_url, headers=headers, data=image_bytes, timeout=20)
        else:
            return jsonify({"error": "No image URL or image data provided"}), 400
    except requests.RequestException as exc:
        return jsonify({"error": f"Request to Azure AI Vision failed: {exc}"}), 502

    try:
        body = resp.json()
    except ValueError:
        body = {"error": resp.text or "Unexpected response from Azure AI Vision"}

    return jsonify(body), resp.status_code


@app.route("/health")
def health():
    return jsonify({"status": "ok", "configured": bool(VISION_KEY and VISION_ENDPOINT)})


if __name__ == "__main__":
    app.run(debug=True)
