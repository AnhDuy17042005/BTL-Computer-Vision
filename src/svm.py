"""
    Các thao tác SVM: train từ đặc trưng, dự đoán nhãn số và lưu/nạp model.
"""

from pathlib import Path
import json

import cv2


def train_svm(features, targets, c=2.5):
    """
        Train OpenCV SVM tuyến tính; mỗi hàng đặc trưng ứng với một nhãn số.
        C điều khiển mức phạt lỗi, mặc định là 2.5.
    """
    if c <= 0:
        raise ValueError("C phải dương")
    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.setC(float(c))
    model.train(features, cv2.ml.ROW_SAMPLE, targets)
    return model


def predict_svm(model, features):
    """
        Dự đoán nhãn số từ ma trận đặc trưng, giữ thứ tự các mẫu đầu vào.
    """
    if model is None:
        raise RuntimeError("Chưa train hoặc nạp model SVM")
    _, predictions = model.predict(features)
    return predictions.ravel()


def save_svm(model, path, metadata):
    """
        Lưu model XML và metadata đi kèm tại đường dẫn có thêm đuôi .json.
    """
    if model is None:
        raise RuntimeError("Không thể lưu model chưa train")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    path.with_suffix(path.suffix+".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def load_svm(path, feature_size):
    """
        Nạp model và metadata; kiểm tra model đã train và đúng số đặc trưng.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {path}")
    metadata = json.loads(path.with_suffix(path.suffix+".json").read_text(encoding="utf-8"))
    model = cv2.ml.SVM_load(str(path))
    if not model.isTrained() or model.getVarCount() != feature_size:
        raise ValueError("Model SVM chưa train hoặc sai kích thước đặc trưng")
    return model, metadata
