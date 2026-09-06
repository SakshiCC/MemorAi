import cv2

from config import CAMERA_INDEX
from utils.constants import EXIT_KEYS


def main() -> None:
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {CAMERA_INDEX}."
        )

    print("Webcam opened. Press Q or Esc to exit.")

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("Could not read webcam frame.")
                break

            cv2.imshow("MemorAi Webcam Test", frame)

            if (cv2.waitKey(1) & 0xFF) in EXIT_KEYS:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
