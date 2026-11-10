"""
    Tách dòng/ký tự bằng threshold, projection và connected components.
"""

from dataclasses import dataclass, field
import cv2
import numpy as np

CHAR_SIZE = (32, 48)  # width, height


@dataclass(frozen=True)
class SegmentConfig:
    """
        Chỉ định số dòng hoặc tự suy ra từ tỷ lệ biển; chọn otsu/adaptive.
    """
    lines: int | None = None
    threshold: str = "adaptive"


@dataclass
class Character:
    """
        Ký tự chuẩn hóa và bbox trên biển đã nắn; chỉ số dòng bắt đầu từ 0.
    """
    image: np.ndarray
    bbox: tuple
    line: int


@dataclass
class SegmentationResult:
    """
        Ký tự được sắp theo thứ tự đọc, cùng crop dòng và ảnh kiểm tra.
    """
    characters: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def normalize_character(image):
    """
        Cắt sát phần chữ trắng, giữ tỷ lệ rồi đặt giữa canvas đen 32×48.
    """
    if image is None or image.size == 0 or image.ndim != 2:
        raise ValueError("Ký tự phải là ảnh xám không rỗng")
    canvas = np.zeros((CHAR_SIZE[1], CHAR_SIZE[0]), np.uint8)
    points = cv2.findNonZero(image)

    if points is None:
        return canvas
    
    x, y, w, h = cv2.boundingRect(points)
    scale = min((CHAR_SIZE[0]-4)/w, (CHAR_SIZE[1]-4)/h)
    resized = cv2.resize(image[y:y+h, x:x+w], (max(1, round(w*scale)), max(1, round(h*scale))),
                         interpolation=cv2.INTER_NEAREST)
    left = (CHAR_SIZE[0]-resized.shape[1])//2
    top = (CHAR_SIZE[1]-resized.shape[0])//2
    canvas[top:top+resized.shape[0], left:left+resized.shape[1]] = resized
    return canvas


def threshold_plate(plate, method):
    """
        Đổi biển thành chữ trắng trên nền đen; xóa mép ngoài để giảm viền biển.
    """
    if plate is None or plate.size == 0:
        raise ValueError("Ảnh biển không được rỗng")
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if plate.ndim == 3 else plate.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 7)
    else:
        raise ValueError("threshold phải là otsu hoặc adaptive")
    border = max(2, round(min(binary.shape)*0.025))
    binary[:border] = binary[-border:] = 0
    binary[:, :border] = binary[:, -border:] = 0
    # Viền còn sót thường chạm mép vùng trong; xóa cả thành phần đó.
    _, components, stats, _ = cv2.connectedComponentsWithStats(binary)

    for component, (x, y, w, h, _) in enumerate(stats[1:], 1):
        if x <= border or y <= border or x+w >= binary.shape[1]-border or y+h >= binary.shape[0]-border:
            binary[components == component] = 0
    return binary


def split_lines(binary, line_count=None):
    """
        Tìm hàng ít pixel chữ nhất ở giữa biển để chia hai dòng.
    """
    h, w = binary.shape
    count = line_count if line_count is not None else (2 if w/h < 2.2 else 1)
    
    if count == 1:
        return [(0, h)]
    if count != 2 or h < 10:
        raise ValueError("Chỉ hỗ trợ biển một hoặc hai dòng, ảnh đủ lớn")
    
    projection = (binary > 0).sum(axis=1).astype(float)
    projection = np.convolve(projection, np.ones(5)/5, mode="same")
    lo, hi = int(h*0.35), int(h*0.65)
    middle = lo+int(np.argmin(projection[lo:hi]))
    return [(0, middle), (middle, h)]


def extract_characters(binary, start, end, line_index):
    """
        Lấy các thành phần có chiều cao/tỷ lệ giống chữ; loại chấm và gạch ngang.
    """
    line = binary[start:end]
    _, components, stats, _ = cv2.connectedComponentsWithStats(line)
    characters = []
    
    for component, (x, y, w, h, area) in enumerate(stats[1:], 1):
        x, y, w, h, area = map(int, (x, y, w, h, area))
        if h < 0.4*(end-start) or w > 0.3*binary.shape[1] or w < max(3, binary.shape[1]*0.025) or area < 12 or h/w < 0.9:
            continue
        crop = np.where(components[y:y+h, x:x+w] == component, 255, 0).astype(np.uint8)
        characters.append(Character(normalize_character(crop), (x, y+start, w, h), line_index))
    return sorted(characters, key=lambda char: char.bbox[0])


def segment_chars(plate, config=None):
    """
        Threshold → chia dòng → lấy ký tự → trả thứ tự trên/dưới, trái/phải.
    """
    config = config or SegmentConfig()
    binary = threshold_plate(plate, config.threshold)
    result = SegmentationResult(debug={"threshold": binary})
    
    for index, (start, end) in enumerate(split_lines(binary, config.lines)):
        result.lines.append(binary[start:end].copy())
        result.characters.extend(extract_characters(binary, start, end, index))
    annotated = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    for char in result.characters:
        x, y, w, h = char.bbox
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 1)
    result.debug["characters"] = annotated
    return result
