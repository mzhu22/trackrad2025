import os

Frames = tuple[str, int]  # (patient_id, sequence_number)

TOKEN = os.getenv("HF_TOKEN")
