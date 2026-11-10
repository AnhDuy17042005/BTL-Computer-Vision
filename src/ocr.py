"""
    OCR: trích HOG, gọi module SVM và ghép/sửa chuỗi ký tự theo vị trí.
"""

from dataclasses import dataclass
from collections import Counter
import re
import cv2
import numpy as np

from .segment import CHAR_SIZE
from .svm import train_svm, predict_svm, save_svm, load_svm

HOG_WINDOW = CHAR_SIZE
HOG_BLOCK = (16, 16)
HOG_STRIDE = (8, 8)
HOG_CELL = (8, 8)
HOG_BINS = 9
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PLATE_PATTERN = re.compile(r"[0-9]{2}[A-Z][A-Z0-9]?[0-9]{4,5}")
TO_DIGIT = str.maketrans({"O": "0", "B": "8", "I": "1", "S": "5"})
TO_LETTER = str.maketrans({"8": "B", "5": "S"})

@dataclass
class RecognitionResult:
    """
        Chuỗi gốc, chuỗi sau chuẩn hóa và cờ đúng format (không phải độ tin cậy).
    """
    raw_text: str
    text: str
    valid: bool


def normalize_text(text):
    """
        Chỉ giữ chữ hoa và số, bỏ dấu chấm/gạch/khoảng trắng.
    """
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def postprocess(raw_text):
    """
        Sửa nhầm chữ/số ở vùng có định dạng rõ; giữ chuỗi gốc để đối chiếu.
    """
    lines = [normalize_text(line) for line in raw_text.splitlines() if line.strip()]
    text = normalize_text(raw_text)

    if len(lines) == 2 and len(lines[0]) in (3, 4) and len(lines[1]) in (4, 5):
        top, bottom = lines
        text = top[:2].translate(TO_DIGIT) + top[2:3].translate(TO_LETTER) + top[3:]
        text += bottom.translate(TO_DIGIT)
    elif len(text) in (7, 8, 9):
        # Chuỗi 9 ký tự có 5 chữ số cuối; chuỗi 8 ký tự có thể có 4 hoặc 5 số.
        digits = 5 if len(text) == 9 else 4
        text = text[:2].translate(TO_DIGIT) + text[2:3].translate(TO_LETTER) + text[3:-digits] + text[-digits:].translate(TO_DIGIT)
    return RecognitionResult(raw_text, text, PLATE_PATTERN.fullmatch(text) is not None)


def augment_character(image):
    """
        Tăng mẫu TRAIN bằng dịch nhẹ và làm nét chữ dày/mỏng; không lấy mẫu test.
    """
    variants = [image]

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        variants.append(cv2.warpAffine(image, matrix, CHAR_SIZE, borderValue=0))
    kernel = np.ones((2, 2), np.uint8)
    variants.append(cv2.dilate(image, kernel))
    variants.append(cv2.erode(image, kernel))
    return [variant for variant in variants if np.count_nonzero(variant)]


class SVMRecognizer:
    """
        Một model OpenCV SVM phân loại ký tự; model không đọc nhãn ảnh lúc dự đoán.
    """

    def __init__(self):
        """
            Khởi tạo bộ HOG; model chỉ có sau khi train hoặc load.
        """
        self.hog = cv2.HOGDescriptor(HOG_WINDOW, HOG_BLOCK, HOG_STRIDE, HOG_CELL, HOG_BINS)
        self.model = None
        self.metadata = {}

    def features(self, images):
        """
            Đổi danh sách ảnh ký tự thành ma trận đặc trưng HOG float32.
        """
        vectors = []
        for image in images:
            if image is None or image.dtype != np.uint8 or image.shape != (48, 32):
                raise ValueError("Ảnh ký tự phải là uint8, shape (48,32)")
            # Crop đã chuẩn hóa từ segment.py; không căn giữa lại để giữ augmentation dịch.
            vectors.append(self.hog.compute(image).ravel())
        return np.asarray(vectors, dtype=np.float32)

    def train(self, images, labels, c=2.5, augment=True):
        """
            Train SVM tuyến tính từ các ký tự đã duyệt; mỗi nhãn là một chữ/số.
            c điều khiển mức phạt lỗi. Giữ kernel LINEAR để baseline đơn giản.
        """
        if not images or len(images) != len(labels) or len(set(labels)) < 2:
            raise ValueError("Cần ảnh/nhãn cùng số lượng và ít nhất hai lớp ký tự")
        if any(len(label) != 1 or label not in ALPHABET for label in labels):
            raise ValueError("Nhãn phải là một ký tự A-Z hoặc 0-9")
        if c <= 0:
            raise ValueError("C phải dương")
        base_features = self.features(images)  # kiểm tra đầu vào trước augmentation
        train_images, train_labels = [], []

        for image, label in zip(images, labels):
            variants = augment_character(image) if augment else [image]
            train_images.extend(variants)
            train_labels.extend([label]*len(variants))
        features = self.features(train_images) if augment else base_features
        targets = np.array([ord(label) for label in train_labels], np.int32)
        self.model = train_svm(features, targets, c)
        self.metadata = {"version": 1, "algorithm": "HOG + OpenCV linear SVM", "c": c,
                         "char_size": list(CHAR_SIZE), "feature_size": int(features.shape[1]),
                         "hog": {"block": list(HOG_BLOCK), "stride": list(HOG_STRIDE),
                                 "cell": list(HOG_CELL), "bins": HOG_BINS},
                         "classes": sorted(set(labels)), "class_counts": dict(sorted(Counter(labels).items())),
                         "original_samples": len(images), "augmented_samples": len(train_images)}

    def predict(self, images):
        """
            Dự đoán từng ký tự; thứ tự kết quả giống thứ tự ảnh đầu vào.
        """
        if self.model is None:
            raise RuntimeError("Chưa train hoặc nạp model SVM")
        if not images:
            return []
        predictions = predict_svm(self.model, self.features(images))
        return [chr(int(value)) for value in predictions]

    def recognize(self, segmentation):
        """
            Nhận kết quả tách ký tự, ghép theo dòng rồi áp dụng hậu xử lý.
        """
        predictions = self.predict([char.image for char in segmentation.characters])
        lines = [[] for _ in segmentation.lines]
        for char, label in zip(segmentation.characters, predictions):
            lines[char.line].append(label)
        return postprocess("\n".join("".join(line) for line in lines))

    def save(self, path):
        """
            Lưu model XML và file .xml.json ghi cấu hình HOG/lớp/nguồn train.
        """
        save_svm(self.model, path, self.metadata)

    @classmethod
    def load(cls, path):
        """
            Nạp model và kiểm tra metadata đúng HOG hiện tại trước khi nhận dạng.
        """
        result = cls()
        model, metadata = load_svm(path, result.hog.getDescriptorSize())
        expected_hog = {"block": list(HOG_BLOCK), "stride": list(HOG_STRIDE),
                        "cell": list(HOG_CELL), "bins": HOG_BINS}

        if (metadata.get("version") != 1 or metadata.get("char_size") != list(CHAR_SIZE)
                or metadata.get("hog") != expected_hog):
            raise ValueError("Metadata model không tương thích với chuẩn hóa/HOG hiện tại")
        result.model = model
        result.metadata = metadata
        return result
