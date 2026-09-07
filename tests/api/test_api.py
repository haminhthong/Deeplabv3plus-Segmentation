"""Unit tests for FastAPI endpoints."""

from __future__ import annotations

import io
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

from vocseg.api.app import app, set_predictor
from vocseg.inference.predictor import Predictor


class DummyModel(nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        logits = torch.zeros((b, 21, h, w), dtype=torch.float32)
        logits[:, 15, :, :] = 5.0  # Lớp Person
        return logits


@pytest.fixture(autouse=True)
def inject_dummy():
    predictor = Predictor.__new__(Predictor)
    predictor.device = torch.device("cpu")
    predictor.image_size = 320
    predictor.checkpoint_meta = {"model_version": "1.0.0"}
    predictor.model = DummyModel()
    set_predictor(predictor)
    yield


def test_api_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "device" in data


def test_api_segment_valid_image():
    client = TestClient(app)
    img = Image.new("RGB", (320, 240), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/segment",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["width"] == 320
    assert data["height"] == 240
    assert data["model_version"] == "1.0.0"
    assert "latency_ms" in data
    assert "mask_png_base64" in data
    assert data["mask_png_base64"].startswith("data:image/png;base64,")


def test_api_segment_invalid_file():
    client = TestClient(app)
    buf = io.BytesIO(b"not an image file")
    response = client.post(
        "/segment",
        files={"file": ("test.txt", buf, "text/plain")},
    )
    assert response.status_code == 400
