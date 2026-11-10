"""
    Phát hiện biển bằng ngưỡng sáng/cạnh, contour và quy tắc hình học đơn giản.
"""

from dataclasses import dataclass, field
import cv2
import numpy as np

from .segment import SegmentationResult, segment_chars


@dataclass(frozen=True)
class DetectorConfig:
    """
        Cấu hình ROI theo tỷ lệ ảnh; chọn edge, bright hoặc both.
    """
    method: str = "both"
    scale: float = 2.0
    roi: tuple = (0.0, 0.0, 1.0, 1.0)
    timestamp_region: tuple | None = (0.55, 0.94, 1.0, 1.0)


@dataclass
class DetectionResult:
    """
        Biển và bbox trên ảnh gốc; giữ kết quả tách ký tự để dùng lại.
    """
    plate: np.ndarray | None = None
    bbox: tuple | None = None
    debug: dict = field(default_factory=dict)
    segmentation: SegmentationResult | None = None


def validate_image(image):
    """
        Kiểm tra ảnh BGR uint8 trước khi đưa vào OpenCV.
    """
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
        raise ValueError("Cần ảnh BGR uint8 không rỗng, shape (H,W,3)")


def pixel_region(region, width, height):
    """
        Đổi (trái,trên,phải,dưới) từ tỷ lệ [0,1] thành tọa độ pixel.
    """
    left, top, right, bottom = region
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("ROI phải thỏa 0 <= trái < phải <= 1 và 0 <= trên < dưới <= 1")
    return int(left*width), int(top*height), int(right*width), int(bottom*height)


def preprocess(image, config):
    """
        Che timestamp, cắt ROI, phóng ảnh, lọc nhiễu và tăng tương phản CLAHE.
        Trả ảnh màu/xám và phép đổi tọa độ về ảnh gốc. Không sửa ảnh đầu vào.
    """
    validate_image(image)
    if not np.isfinite(config.scale) or config.scale <= 0:
        raise ValueError("Hệ số phóng ảnh phải dương")
    height, width = image.shape[:2]
    x0, y0, x1, y1 = pixel_region(config.roi, width, height)
    masked = image.copy()

    if config.timestamp_region is not None:
        a, b, c, d = pixel_region(config.timestamp_region, width, height)
        masked[b:d, a:c] = 0
    crop = masked[y0:y1, x0:x1]

    if crop.size == 0:
        raise ValueError("ROI quá nhỏ")
    size = (max(1, round(crop.shape[1]*config.scale)), max(1, round(crop.shape[0]*config.scale)))
    color = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    scales = np.float32([size[0]/crop.shape[1], size[1]/crop.shape[0]])
    return color, gray, np.float32([x0, y0]), scales


def make_masks(color, gray, method):
    """
        Tạo mặt nạ cạnh dọc và/hoặc vùng sáng rồi nối các vùng gần nhau.
    """
    if method not in {"edge", "bright", "both"}:
        raise ValueError("method phải là edge, bright hoặc both")
    masks = {}

    if method in {"edge", "both"}:
        sobel = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0))
        _, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks["edge_morphology"] = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 17), np.uint8))
    
    if method in {"bright", "both"}:
        brightness = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        for level in (0, 120, 160, 200):
            mode = cv2.THRESH_BINARY + (cv2.THRESH_OTSU if level == 0 else 0)
            _, binary = cv2.threshold(brightness, level, 255, mode)
            masks[f"bright_{level}"] = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return masks


def candidate_score(contour, gray, saturation, edge):
    """
        Loại vùng quá nhỏ/lớn, quá đỏ hoặc ít cạnh; chấm điểm vùng giống biển.
        Điểm chỉ dùng để xếp hạng, không phải xác suất phát hiện đúng.
    """
    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    if min(rw, rh) < 10:
        return None
    
    x, y, w, h = cv2.boundingRect(contour)
    area_ratio = rw*rh/gray.size
    extent = cv2.contourArea(contour)/(rw*rh)

    if not (0.002 <= area_ratio <= 0.15 and 0.65 <= w/h <= 6 and extent >= 0.4):
        return None
    sat = float(saturation[y:y+h, x:x+w].mean())
    density = np.count_nonzero(edge[y:y+h, x:x+w])/(w*h)

    if sat > 110 or not 0.035 <= density <= 0.5:
        return None
    
    _, ink = cv2.threshold(gray[y:y+h, x:x+w], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    count = sum(0.18*h < ch < 0.9*h and 0.015*w < cw < 0.3*w and area > 8
                for _, _, cw, ch, area in stats[1:])
    score = 2*min(count, 10)/10 + extent + min(density/0.15, 1) + (1-sat/255)
    return float(score), cv2.boxPoints(rect)


def find_candidates(masks, color, gray, edge):
    """
        Duyệt contour, giữ ứng viên hợp lệ và sắp điểm giảm dần.
    """
    saturation = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[:, :, 1]
    candidates = []
    for mask in masks.values():
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            candidate = candidate_score(contour, gray, saturation, edge)
            if candidate is not None:
                candidates.append(candidate)

    unique = []
    for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        box = cv2.boundingRect(candidate[1])
        if not any(box_overlap(box, cv2.boundingRect(other[1])) > 0.8 for other in unique):
            unique.append(candidate)
    return unique


def box_overlap(first, second):
    """
        Tính IoU hai rectangle để bỏ ứng viên trùng nhau.
    """
    x, y, w, h = first
    a, b, c, d = second
    overlap = max(0, min(x+w, a+c)-max(x, a))*max(0, min(y+h, b+d)-max(y, b))
    return overlap/(w*h+c*d-overlap)


def order_corners(points):
    """
        Sắp góc theo thứ tự trên-trái, trên-phải, dưới-phải, dưới-trái.
    """
    points = np.asarray(points, np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0])
    ordered = points[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)


def rectify_plate(image, corners):
    """
        Nắn biển nghiêng nhẹ, giữ tỷ lệ; chiều rộng 200 (hai dòng) hoặc 320 (một dòng).
    """
    points = order_corners(corners)
    tl, tr, br, bl = points
    width = max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl))
    height = max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr))

    if min(width, height) < 2:
        raise ValueError("Bốn góc không tạo được biển hợp lệ")
    
    out_w = 320 if width/height > 2.2 else 200
    out_h = max(32, round(out_w*height/width))
    target = np.float32([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]])
    matrix = cv2.getPerspectiveTransform(points, target)
    return cv2.warpPerspective(image, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def choose_candidate(candidates, image, offset, scales):
    """
        Ưu tiên ứng viên có 7-9 ký tự và bố cục dòng hợp lý, không dùng nhãn thật.
        Kiểm tra tối đa 20 vùng để tránh chọn đèn xe hoặc chỉ một dòng của biển.
    """
    choices = []
    for score, box in candidates[:20]:
        corners = box/scales + offset
        corners[:, 0] = np.clip(corners[:, 0], 0, image.shape[1]-1)
        corners[:, 1] = np.clip(corners[:, 1], 0, image.shape[0]-1)
        plate = rectify_plate(image, corners)
        segmented = segment_chars(plate)
        counts = [sum(c.line == i for c in segmented.characters) for i in range(len(segmented.lines))]
        
        if len(counts) == 2:
            layout = 3 <= counts[0] <= 4 and 4 <= counts[1] <= 5
        else:
            layout = 7 <= sum(counts) <= 9
        if not layout:
            continue
        
        layout_bonus = 2.0 if len(counts) == 2 and counts[0] == 4 else 0.0
        choices.append((score+layout_bonus, corners, plate, segmented))
    return max(choices, key=lambda item: item[0]) if choices else None


def detect_plate(image, config=None):
    """
        Tiền xử lý, chọn ứng viên rồi trả ảnh biển, bbox và ký tự đã tách.
    """
    config = config or DetectorConfig()
    color, gray, offset, scales = preprocess(image, config)
    edge = cv2.Canny(gray, 70, 180)
    masks = make_masks(color, gray, config.method)
    debug = {"gray": gray, "edge": edge, **masks}
    candidates = find_candidates(masks, color, gray, edge)
    chosen = choose_candidate(candidates, image, offset, scales)
    
    if chosen is None:
        return DetectionResult(debug=debug)
    
    _, corners, plate, segmented = chosen
    x0, y0 = np.floor(corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int)+1
    bbox = (int(x0), int(y0), int(x1-x0), int(y1-y0))
    annotated = image.copy()
    cv2.polylines(annotated, [np.rint(corners).astype(np.int32)], True, (0, 255, 0), 2)
    debug.update({"detection": annotated, "plate": plate})
    return DetectionResult(plate, bbox, debug, segmented)
