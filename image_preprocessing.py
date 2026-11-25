import cv2
import matplotlib.pyplot as plt


def enhance_frame(frame):
    """Зменшення шуму та вирівнювання гістограми яскравості (CLAHE)."""
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enh = clahe.apply(l_ch)

    merged = cv2.merge((l_enh, a_ch, b_ch))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def save_histogram_comparison(orig_frame, enh_frame, out_path, title="Гістограма яскравості"):
    """Зберігає порівняльну гістограму яскравості до і після обробки."""
    gray_orig = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
    gray_enh = cv2.cvtColor(enh_frame, cv2.COLOR_BGR2GRAY)

    hist_orig = cv2.calcHist([gray_orig], [0], None, [256], [0, 256])
    hist_enh = cv2.calcHist([gray_enh], [0], None, [256], [0, 256])

    plt.figure(figsize=(9, 4))
    plt.plot(hist_orig, color="gray", label="Оригінальний кадр")
    plt.plot(hist_enh, color="blue", label="Після CLAHE")
    plt.title(title)
    plt.xlabel("Яскравість (0-255)")
    plt.ylabel("Кількість пікселів")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
