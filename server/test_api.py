from fastapi.testclient import TestClient
import sys
import os
from PIL import Image
import io

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.main import app
from src.data.split import get_test_paths

client = TestClient(app)

def create_mock_photo(face_path):
    face = Image.open(face_path).convert("RGB")
    face = face.resize((150, 150), Image.BILINEAR)
    photo = Image.new('RGB', (300, 300), color=(200, 200, 200))
    photo.paste(face, (75, 75))
    return photo

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "V3-B Residual CNN"
    print("GET /health passed.")

def test_predict_success():
    test_paths = get_test_paths()
    sample_image_path = test_paths['happy'][0]
    
    photo = create_mock_photo(sample_image_path)
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    photo.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "predicted_class" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "bbox" in data
    
    probs = data["probabilities"]
    assert len(probs) == 7
    
    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) < 1e-5
    
    assert data["confidence"] == probs[data["predicted_class"]]
    
    print("POST /predict (success) passed.")

def test_predict_no_face():
    # Solid black image
    photo = Image.new('RGB', (200, 200), color='black')
    
    img_byte_arr = io.BytesIO()
    photo.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )
    
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "NO_FACE_DETECTED"
    print("POST /predict (no face) passed.")
    
def test_predict_invalid():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    print("POST /predict (invalid) passed.")

if __name__ == "__main__":
    print("Running API tests...")
    test_health()
    test_predict_success()
    test_predict_no_face()
    test_predict_invalid()
    print("All API tests passed!")
