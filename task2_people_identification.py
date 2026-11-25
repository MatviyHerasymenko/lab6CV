import argparse
import os
import time
import cv2
import numpy as np

from image_preprocessing import enhance_frame, save_histogram_comparison


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO = os.path.join(BASE_DIR, "videos", "people_faces.mp4")
PROTOTXT_PATH = os.path.join(BASE_DIR, "models", "MobileNetSSD_deploy.prototxt.txt")
MODEL_PATH = os.path.join(BASE_DIR, "models", "MobileNetSSD_deploy.caffemodel")
FACE_CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_eye.xml")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]


def main():
    parser = argparse.ArgumentParser(description="Ієрархічна ідентифікація людей")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Шлях до відео")
    parser.add_argument("--confidence", type=float, default=0.45, help="Поріг впевненості")
    parser.add_argument("--no-show", action="store_true", help="Запуск без вікна перегляду")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    if face_cascade.empty() or eye_cascade.empty():
        raise IOError("Не вдалося завантажити каскади Хаара.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Не вдалося відкрити відео: {args.video}")

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 12.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = os.path.join(OUTPUT_DIR, "people_detected.mp4")
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (width, height))

    frame_idx = 0
    total_persons = 0
    total_faces = 0
    total_eyes = 0
    saved_screenshots = 0
    hist_saved = False
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        enhanced = enhance_frame(frame)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        if not hist_saved and frame_idx == 20:
            save_histogram_comparison(
                frame, enhanced,
                os.path.join(OUTPUT_DIR, "people_histogram.png"),
                title="Гістограма яскравості (відео з людьми)"
            )
            hist_saved = True

        # Рівень 1: детекція людини через MobileNetSSD
        blob = cv2.dnn.blobFromImage(cv2.resize(enhanced, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        persons_in_frame = 0
        faces_in_frame = 0

        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < args.confidence or CLASSES[int(detections[0, 0, i, 1])] != "person":
                continue

            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            startX, startY, endX, endY = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(width - 1, endX), min(height - 1, endY)
            if endX <= startX or endY <= startY:
                continue

            persons_in_frame += 1
            total_persons += 1
            cv2.rectangle(enhanced, (startX, startY), (endX, endY), (255, 160, 0), 2)
            cv2.putText(enhanced, f"Person: {conf * 100:.1f}%", (startX, max(15, startY - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2)

            # Рівень 2: пошук обличчя в ROI людини
            person_roi_gray = gray[startY:endY, startX:endX]
            person_roi_color = enhanced[startY:endY, startX:endX]
            faces = face_cascade.detectMultiScale(person_roi_gray, scaleFactor=1.15, minNeighbors=4, minSize=(24, 24))

            for (fx, fy, fw, fh) in faces:
                faces_in_frame += 1
                total_faces += 1
                cv2.rectangle(person_roi_color, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

                # Рівень 3: пошук очей в ROI обличчя
                face_roi_gray = person_roi_gray[fy:fy + fh, fx:fx + fw]
                face_roi_color = person_roi_color[fy:fy + fh, fx:fx + fw]
                eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(8, 8))

                for (ex, ey, ew, eh) in eyes:
                    total_eyes += 1
                    center = (int(ex + 0.5 * ew), int(ey + 0.5 * eh))
                    radius = max(2, int(0.25 * (ew + eh)))
                    cv2.circle(face_roi_color, center, radius, (0, 0, 255), 2)

        if persons_in_frame > 0 and faces_in_frame > 0 and saved_screenshots < 3 and frame_idx % 40 == 0:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"people_frame_{frame_idx}.jpg"), enhanced)
            saved_screenshots += 1

        cv2.putText(enhanced, f"People: {persons_in_frame} | Faces: {faces_in_frame}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        writer.write(enhanced)

        if not args.no_show:
            cv2.imshow("People Identification", enhanced)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    elapsed = time.time() - start_time
    cap.release()
    writer.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    print(f"Оброблено кадрів: {frame_idx} | Людей: {total_persons} | Облич: {total_faces} | Очей: {total_eyes} | FPS: {frame_idx / elapsed:.2f}")


if __name__ == "__main__":
    main()
