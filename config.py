from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# BASE PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(
    BASE_DIR / ".env"
)

PROJECT_NAME = "MemorAi"


# ============================================================
# DATABASE
# ============================================================

DB_HOST = os.getenv(
    "DB_HOST",
    "localhost",
)

DB_PORT = int(
    os.getenv(
        "DB_PORT",
        "3306",
    )
)

DB_USER = os.getenv(
    "DB_USER",
    "root",
)

DB_PASSWORD = os.getenv(
    "DB_PASSWORD",
    "",
)

DB_NAME = os.getenv(
    "DB_NAME",
    "Memor_ai",
)

DB_POOL_NAME = os.getenv(
    "DB_POOL_NAME",
    "memorai_pool",
)

DB_POOL_SIZE = int(
    os.getenv(
        "DB_POOL_SIZE",
        "5",
    )
)


# ============================================================
# GENERAL PATHS
# ============================================================

MODEL_DIR = (
    BASE_DIR
    / "models"
)

YOLO_MODEL_PATH = Path(
    os.getenv(
        "YOLO_MODEL_PATH",
        str(
            MODEL_DIR
            / "yolo"
            / "yolo11s.pt"
        ),
    )
)

EVIDENCE_DIR = Path(
    os.getenv(
        "EVIDENCE_DIR",
        str(
            BASE_DIR
            / "evidence"
        ),
    )
)

CHROMA_PATH = Path(
    os.getenv(
        "CHROMA_PATH",
        str(
            BASE_DIR
            / "chroma_data"
        ),
    )
)

LOG_DIR = (
    BASE_DIR
    / "logs"
)


# ============================================================
# CREATE GENERAL DIRECTORIES
# ============================================================

for directory in (
    EVIDENCE_DIR,
    CHROMA_PATH,
    LOG_DIR,
):

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# VISION / OBJECT DETECTION
# ============================================================

CAMERA_INDEX = int(
    os.getenv(
        "CAMERA_INDEX",
        "0",
    )
)

YOLO_CONFIDENCE = float(
    os.getenv(
        "YOLO_CONFIDENCE",
        "0.45",
    )
)

CONFIRMATION_FRAMES = int(
    os.getenv(
        "CONFIRMATION_FRAMES",
        "4",
    )
)

DETECTION_COOLDOWN_SECONDS = float(
    os.getenv(
        "DETECTION_COOLDOWN_SECONDS",
        "8",
    )
)


# ============================================================
# IMPORTANT OBJECTS
# ============================================================

IMPORTANT_OBJECTS = {
    item.strip().lower()
    for item in os.getenv(
        "IMPORTANT_OBJECTS",
        (
            "cell phone,"
            "bottle,"
            "cup,"
            "book,"
            "remote,"
            "handbag,"
            "backpack,"
            "keys,"
            "scissors"
        ),
    ).split(",")
    if item.strip()
}


# ============================================================
# FACE RECOGNITION
# ============================================================

FACE_MODEL_DIR = (
    MODEL_DIR
    / "face"
)


# ------------------------------------------------------------
# YuNet Detection Model
# ------------------------------------------------------------

FACE_DETECTION_MODEL_PATH = Path(
    os.getenv(
        "FACE_DETECTION_MODEL_PATH",
        str(
            FACE_MODEL_DIR
            / "face_detection_yunet_2023mar.onnx"
        ),
    )
)


# ------------------------------------------------------------
# SFace Recognition Model
# ------------------------------------------------------------

FACE_RECOGNITION_MODEL_PATH = Path(
    os.getenv(
        "FACE_RECOGNITION_MODEL_PATH",
        str(
            FACE_MODEL_DIR
            / "face_recognition_sface_2021dec.onnx"
        ),
    )
)


# ------------------------------------------------------------
# YuNet Detection Threshold
# ------------------------------------------------------------

FACE_DETECTION_SCORE_THRESHOLD = float(
    os.getenv(
        "FACE_DETECTION_SCORE_THRESHOLD",
        "0.60",
    )
)


# ------------------------------------------------------------
# YuNet NMS Threshold
# ------------------------------------------------------------

FACE_DETECTION_NMS_THRESHOLD = float(
    os.getenv(
        "FACE_DETECTION_NMS_THRESHOLD",
        "0.30",
    )
)


# ------------------------------------------------------------
# SFace Identity Threshold
# ------------------------------------------------------------

FACE_RECOGNITION_COSINE_THRESHOLD = float(
    os.getenv(
        "FACE_RECOGNITION_COSINE_THRESHOLD",
        "0.50",
    )
)


# ------------------------------------------------------------
# Number of strongest samples used per identity
# ------------------------------------------------------------

FACE_RECOGNITION_TOP_K = int(
    os.getenv(
        "FACE_RECOGNITION_TOP_K",
        "3",
    )
)


# ------------------------------------------------------------
# Multi-frame Face Confirmation
# ------------------------------------------------------------

FACE_CONFIRMATION_FRAMES = int(
    os.getenv(
        "FACE_CONFIRMATION_FRAMES",
        "3",
    )
)


# ------------------------------------------------------------
# Encounter Cooldown
# ------------------------------------------------------------

FACE_EVENT_COOLDOWN_SECONDS = float(
    os.getenv(
        "FACE_EVENT_COOLDOWN_SECONDS",
        "120",
    )
)


# ------------------------------------------------------------
# Registration Sample Count
# ------------------------------------------------------------

FACE_REGISTRATION_SAMPLES = int(
    os.getenv(
        "FACE_REGISTRATION_SAMPLES",
        "5",
    )
)


# ============================================================
# PEOPLE PATHS
# ============================================================

PEOPLE_EVIDENCE_DIR = (
    EVIDENCE_DIR
    / "people"
)

REGISTERED_FACE_IMAGE_DIR = (
    BASE_DIR
    / "data"
    / "people"
    / "registered"
)


# ============================================================
# CREATE PEOPLE DIRECTORIES
# ============================================================

for directory in (
    PEOPLE_EVIDENCE_DIR,
    REGISTERED_FACE_IMAGE_DIR,
):

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# MEDICINE VISUAL MEMORY
# ============================================================

MEDICINE_MODEL_DIR = (
    MODEL_DIR
    / "medicine"
)


# ------------------------------------------------------------
# DINOv2 Visual Embedding Model
# ------------------------------------------------------------

MEDICINE_EMBEDDING_MODEL_NAME = os.getenv(
    "MEDICINE_EMBEDDING_MODEL_NAME",
    "facebook/dinov2-small",
)


# ------------------------------------------------------------
# DINOv2 Device
# ------------------------------------------------------------

MEDICINE_EMBEDDING_DEVICE = os.getenv(
    "MEDICINE_EMBEDDING_DEVICE",
    "cpu",
)


# ------------------------------------------------------------
# Registration Samples
# ------------------------------------------------------------

MEDICINE_REGISTRATION_SAMPLES = int(
    os.getenv(
        "MEDICINE_REGISTRATION_SAMPLES",
        "5",
    )
)


# ------------------------------------------------------------
# Medicine Recognition Similarity Threshold
#
# Live webcam calibration:
#
# Registered Omez produced genuine live scores mainly around
# 0.60 - 0.74 under normal camera variation.
#
# 0.75 was too strict and caused valid medicine frames to
# continuously reset the multi-frame confirmation sequence.
#
# We therefore use 0.60 for the current prototype while still
# requiring multiple consecutive accepted frames.
# ------------------------------------------------------------

MEDICINE_RECOGNITION_THRESHOLD = float(
    os.getenv(
        "MEDICINE_RECOGNITION_THRESHOLD",
        "0.60",
    )
)


# ------------------------------------------------------------
# Strongest Reference Samples
# ------------------------------------------------------------

MEDICINE_RECOGNITION_TOP_K = int(
    os.getenv(
        "MEDICINE_RECOGNITION_TOP_K",
        "3",
    )
)


# ------------------------------------------------------------
# Multi-frame Medicine Confirmation
# ------------------------------------------------------------

MEDICINE_CONFIRMATION_FRAMES = int(
    os.getenv(
        "MEDICINE_CONFIRMATION_FRAMES",
        "3",
    )
)


# ------------------------------------------------------------
# Medicine Recognition Cooldown
# ------------------------------------------------------------

MEDICINE_RECOGNITION_COOLDOWN_SECONDS = float(
    os.getenv(
        "MEDICINE_RECOGNITION_COOLDOWN_SECONDS",
        "5",
    )
)


# ============================================================
# MEDICINE PATHS
# ============================================================

REGISTERED_MEDICINE_IMAGE_DIR = (
    BASE_DIR
    / "data"
    / "medicines"
    / "registered"
)

MEDICINE_EVIDENCE_DIR = (
    EVIDENCE_DIR
    / "medicines"
)


# ============================================================
# MEDICATION ACTION / HAND TRACKING
# ============================================================

MEDICINE_HAND_MODEL_PATH = Path(
    os.getenv(
        "MEDICINE_HAND_MODEL_PATH",
        str(
            MEDICINE_MODEL_DIR
            / "hand_landmarker.task"
        ),
    )
)


# ------------------------------------------------------------
# Hand Landmarker Confidence
# ------------------------------------------------------------

MEDICINE_HAND_DETECTION_CONFIDENCE = float(
    os.getenv(
        "MEDICINE_HAND_DETECTION_CONFIDENCE",
        "0.50",
    )
)

MEDICINE_HAND_PRESENCE_CONFIDENCE = float(
    os.getenv(
        "MEDICINE_HAND_PRESENCE_CONFIDENCE",
        "0.50",
    )
)

MEDICINE_HAND_TRACKING_CONFIDENCE = float(
    os.getenv(
        "MEDICINE_HAND_TRACKING_CONFIDENCE",
        "0.50",
    )
)


# ------------------------------------------------------------
# Action State Machine
# ------------------------------------------------------------

MEDICINE_ACTION_WINDOW_SECONDS = float(
    os.getenv(
        "MEDICINE_ACTION_WINDOW_SECONDS",
        "20",
    )
)


# ------------------------------------------------------------
# Mouth Zone
# ------------------------------------------------------------

MEDICINE_ACTION_MOUTH_NEAR_RATIO = float(
    os.getenv(
        "MEDICINE_ACTION_MOUTH_NEAR_RATIO",
        "0.38",
    )
)

MEDICINE_ACTION_MOUTH_EXIT_RATIO = float(
    os.getenv(
        "MEDICINE_ACTION_MOUTH_EXIT_RATIO",
        "0.65",
    )
)


# ------------------------------------------------------------
# Temporal Action Confirmation
# ------------------------------------------------------------

MEDICINE_ACTION_NEAR_FRAMES = int(
    os.getenv(
        "MEDICINE_ACTION_NEAR_FRAMES",
        "2",
    )
)

MEDICINE_ACTION_RETREAT_FRAMES = int(
    os.getenv(
        "MEDICINE_ACTION_RETREAT_FRAMES",
        "2",
    )
)


# ------------------------------------------------------------
# Minimum Action Confidence
# ------------------------------------------------------------

MEDICINE_ACTION_CONFIRMATION_THRESHOLD = float(
    os.getenv(
        "MEDICINE_ACTION_CONFIRMATION_THRESHOLD",
        "0.70",
    )
)


# ============================================================
# CREATE MEDICINE DIRECTORIES
# ============================================================

for directory in (
    MEDICINE_MODEL_DIR,
    REGISTERED_MEDICINE_IMAGE_DIR,
    MEDICINE_EVIDENCE_DIR,
):

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# EMBEDDINGS / CHROMADB
# ============================================================

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "BAAI/bge-small-en-v1.5",
)

CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "memorai_memories",
)

DEFAULT_TOP_K = int(
    os.getenv(
        "DEFAULT_TOP_K",
        "5",
    )
)

MAX_DISTANCE = float(
    os.getenv(
        "MAX_DISTANCE",
        "1.20",
    )
)


# ============================================================
# OLLAMA
# ============================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b",
)


# ============================================================
# WHISPER
# ============================================================

WHISPER_MODEL_SIZE = os.getenv(
    "WHISPER_MODEL_SIZE",
    "small",
)

WHISPER_DEVICE = os.getenv(
    "WHISPER_DEVICE",
    "cpu",
)

WHISPER_COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE",
    "int8",
)