"""
    Điều phối một ảnh: detect → segment → OCR; lưu kết quả cho CLI và GUI.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import json
from time import perf_counter
from typing import Literal

import numpy as np
import cv2

from .detect import DetectionResult, DetectorConfig, detect_plate
from .ocr import RecognitionResult, SVMRecognizer
from .segment import SegmentConfig, SegmentationResult, segment_chars


@dataclass(frozen=True)
class PipelineConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    segmentation: SegmentConfig = field(default_factory=SegmentConfig)


@dataclass
class PipelineResult:
    detection: DetectionResult
    segmentation: SegmentationResult
    recognition: RecognitionResult | None
    status: Literal["no_plate", "no_characters", "no_model", "invalid_text", "ok"]
    elapsed_ms: float
    debug: dict[str, np.ndarray] = field(default_factory=dict)


class PlatePipeline:
    def __init__(self, recognizer: SVMRecognizer | None = None, config: PipelineConfig | None = None):
        """
            Nhận model dùng chung cho nhiều ảnh và cấu hình từng công đoạn.
        """
        self.recognizer = recognizer
        self.config = config or PipelineConfig()

    def process(self, image: np.ndarray) -> PipelineResult:
        """
            Chạy một ảnh; trả trạng thái rõ ràng và ảnh trung gian của từng bước.
        """
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
            raise ValueError("Cần ảnh BGR uint8 không rỗng, shape (H,W,3)")
        
        start = perf_counter()
        detection = detect_plate(image, self.config.detector)
        segmentation = SegmentationResult()
        recognition = None
        status = "no_plate"
        
        if detection.plate is not None:
            segmentation = segment_chars(detection.plate, self.config.segmentation)
            status = "no_characters"
            if segmentation.characters:
                status = "no_model"
                if self.recognizer is not None:
                    recognition = self.recognizer.recognize(segmentation)
                    status = "ok" if recognition.valid else "invalid_text"

        return PipelineResult(detection, segmentation, recognition, status,
                              (perf_counter()-start)*1000,
                              {**detection.debug, **segmentation.debug})


def read_image(path):
    """
        Đọc ảnh BGR; báo lỗi có đường dẫn khi file không đọc được.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return image


def result_summary(result):
    """
        Chuyển kết quả thành dictionary nhẹ để hiển thị hoặc xuất JSON.
    """
    return {"status": result.status, "bbox": result.detection.bbox,
            "character_count": len(result.segmentation.characters),
            "recognition": asdict(result.recognition) if result.recognition else None,
            "elapsed_ms": result.elapsed_ms}


def save_result(result, directory):
    """
        Xuất JSON, CSV, ảnh các bước và crop ký tự vào một thư mục.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary = result_summary(result)
    (directory/"result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

    with (directory/"result.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["status", "plate", "raw_text", "bbox", "elapsed_ms"])
        recognition = result.recognition
        writer.writerow([result.status, recognition.text if recognition else "",
                         recognition.raw_text if recognition else "", result.detection.bbox, result.elapsed_ms])
    
    images = dict(result.debug)
    images.update({f"char_{index:02d}": char.image for index, char in enumerate(result.segmentation.characters)})

    for name, image in images.items():
        if not cv2.imwrite(str(directory/f"{name}.png"), image):
            raise OSError(f"Không ghi được ảnh: {name}")
