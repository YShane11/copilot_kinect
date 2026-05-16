import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.vision.face_recognition_db import FaceRecognitionDB


def main():
    face_db = FaceRecognitionDB(BASE_DIR)
    students = face_db.build_database()

    print(f'Embedding DB built for {len(students)} student(s).')
    for student in students:
        print(
            f"- {student['label']}: "
            f"{student['matched_images']} matched image(s) / {student['image_count']} total image(s)"
        )


if __name__ == '__main__':
    main()
