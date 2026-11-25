import argparse
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

from image_preprocessing import enhance_frame, save_histogram_comparison


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO = os.path.join(BASE_DIR, "videos", "cars_traffic.mp4")
PROTOTXT_PATH = os.path.join(BASE_DIR, "models", "MobileNetSSD_deploy.prototxt.txt")
MODEL_PATH = os.path.join(BASE_DIR, "models", "MobileNetSSD_deploy.caffemodel")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]
VEHICLE_CLASSES = {"car", "bus", "motorbike"}


def extract_car_features(roi_bgr):
    """Векторизація ROI авто: середні HSV-значення та пропорція рамки."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v = cv2.mean(hsv)[:3]
    h, w = roi_bgr.shape[:2]
    aspect = w / float(h) if h > 0 else 1.0
    return [mean_h, mean_s, mean_v, aspect]


def save_dendrogram(features, out_path):
    """Ієрархічна кластеризація (Ward) та збереження дендрограми."""
    if len(features) < 3:
        return

    sample = np.array(features[:40])
    scaled = StandardScaler().fit_transform(sample)

    clustering = AgglomerativeClustering(n_clusters=min(3, len(sample)), linkage="ward")
    labels = clustering.fit_predict(scaled)
    print(f"[INFO] Кластеризація (Ward): розподіл по кластерах = {np.bincount(labels)}")

    Z = linkage(scaled, method="ward")
    plt.figure(figsize=(10, 5))
    dendrogram(Z, leaf_rotation=90, leaf_font_size=8)
    plt.title("Дендрограма ієрархічної кластеризації автівок")
    plt.xlabel("Індекс виявленого об'єкта")
    plt.ylabel("Відстань (Ward)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Ідентифікація автівок у відеопотоці")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Шлях до відео")
    parser.add_argument("--confidence", type=float, default=0.35, help="Поріг впевненості")
    parser.add_argument("--no-show", action="store_true", help="Запуск без вікна перегляду")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Не вдалося відкрити відео: {args.video}")

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 12.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = os.path.join(OUTPUT_DIR, "cars_detected.mp4")
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (width, height))

    frame_idx = 0
    total_detections = 0
    frames_with_cars = 0
    car_features = []
    saved_screenshots = 0
    hist_saved = False
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        enhanced = enhance_frame(frame)

        if not hist_saved and frame_idx == 10:
            save_histogram_comparison(
                frame, enhanced,
                os.path.join(OUTPUT_DIR, "cars_histogram.png"),
                title="Гістограма яскравості (відео з автівками)"
            )
            hist_saved = True

        # Детекція об'єктів через MobileNetSSD
        blob = cv2.dnn.blobFromImage(cv2.resize(enhanced, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        cars_in_frame = 0
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < args.confidence:
                continue

            cls_name = CLASSES[int(detections[0, 0, i, 1])]
            if cls_name not in VEHICLE_CLASSES:
                continue

            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            startX, startY, endX, endY = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(width - 1, endX), min(height - 1, endY)
            if endX <= startX or endY <= startY:
                continue

            cars_in_frame += 1
            total_detections += 1
            car_features.append(extract_car_features(enhanced[startY:endY, startX:endX]))

            label = f"Vehicle -> {cls_name}: {conf * 100:.1f}%"
            cv2.rectangle(enhanced, (startX, startY), (endX, endY), (0, 200, 255), 2)
            y_text = startY - 10 if startY - 10 > 15 else startY + 20
            cv2.putText(enhanced, label, (startX, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        if cars_in_frame > 0:
            frames_with_cars += 1
            if saved_screenshots < 3 and frame_idx % 25 == 0:
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"cars_frame_{frame_idx}.jpg"), enhanced)
                saved_screenshots += 1

        cv2.putText(enhanced, f"Vehicles: {cars_in_frame}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        writer.write(enhanced)

        if not args.no_show:
            cv2.imshow("Cars Identification", enhanced)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    elapsed = time.time() - start_time
    cap.release()
    writer.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    save_dendrogram(car_features, os.path.join(OUTPUT_DIR, "cars_dendrogram.png"))

    print(f"Оброблено кадрів: {frame_idx} | Кадрів з авто: {frames_with_cars} | Детекцій: {total_detections} | FPS: {frame_idx / elapsed:.2f}")


if __name__ == "__main__":
    main()
