"""
    Nhóm E: audit, chia tập theo biển số, chuẩn bị ký tự và đánh giá.

    Ảnh toàn cảnh + nhãn chuỗi là dữ liệu nguồn, chưa phải mẫu ký tự đưa trực tiếp
    vào SVM. Audit tự động không chứng minh nhãn đúng với chữ trên ảnh.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import random
import re

import cv2
import numpy as np

from .detect import detect_plate
from .segment import segment_chars

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
PLATE_PATTERN = re.compile(r"[0-9]{2}[A-Z][A-Z0-9]?[0-9]{4,5}")


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    image_path: Path
    label_path: Path
    text: str


def audit_dataset(root: str | Path) -> dict:
    """
        Kiểm tra tên cặp, ảnh đọc được và format nhãn; không tự sửa dữ liệu.
    """
    root = Path(root)
    indexes = []
    
    for folder, extensions in (("images", IMAGE_EXTENSIONS), ("labels", {".txt"})):
        index = defaultdict(list)
        for path in sorted((root/folder).iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                index[path.stem].append(path)
        indexes.append(index)
    images, labels = indexes
    errors = []
    
    for stem in sorted(images.keys()-labels.keys()):
        errors.append(f"Thiếu nhãn: {stem}")
    for stem in sorted(labels.keys()-images.keys()):
        errors.append(f"Thiếu ảnh: {stem}")
    for folder, index in zip(("images", "labels"), indexes):
        for stem, paths in index.items():
            if len(paths) != 1:
                errors.append(f"Trùng tên mẫu trong {folder}: {stem}")
    dimensions = Counter()
    texts = {}
    
    for paths in images.values():
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                errors.append(f"Ảnh không đọc được: {path.name}")
            else:
                dimensions[f"{image.shape[1]}x{image.shape[0]}"] += 1
    for stem, paths in labels.items():
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError) as exc:
                errors.append(f"Nhãn không đọc được: {path.name}: {exc}")
                continue
            if PLATE_PATTERN.fullmatch(text) is None:
                errors.append(f"Nhãn sai định dạng: {path.name}: {text!r}")
            else:
                texts[stem] = text
    if not images:
        errors.append("Không có ảnh")
    return {"ok": not errors, "image_count": sum(map(len, images.values())),
            "label_count": sum(map(len, labels.values())),
            "paired_count": len(images.keys() & labels.keys()),
            "unique_plates": len({text for stem, text in texts.items() if stem in images}),
            "dimensions": dict(dimensions), "errors": errors}


def load_dataset(root: str | Path) -> list[DatasetSample]:
    """
        Đọc danh sách mẫu đã qua audit, giữ đường dẫn ảnh/nhãn và chuỗi thật.
    """
    root = Path(root)
    report = audit_dataset(root)
    
    if not report["ok"]:
        raise ValueError("Dataset không hợp lệ: " + "; ".join(report["errors"]))
    labels = {p.stem: p for p in (root/"labels").iterdir() if p.suffix.lower() == ".txt"}
    return [DatasetSample(p.stem, p, labels[p.stem], labels[p.stem].read_text(encoding="utf-8").strip())
            for p in sorted((root/"images").iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def split_dataset(samples: list[DatasetSample], seed: int = 42) -> dict:
    """
        70/15/15 theo biển số; các ảnh cùng biển luôn thuộc cùng một tập.
    """
    groups = sorted({s.text for s in samples})
    
    if len(groups) < 7:
        raise ValueError("Cần ít nhất 7 biển khác nhau để chia train/validation/test")
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("sample_id phải duy nhất")
    random.Random(seed).shuffle(groups)
    count = max(1, round(len(groups)*0.15))
    val, test = set(groups[:count]), set(groups[count:2*count])
    subsets = {"train": [], "validation": [], "test": []}
    
    for sample in sorted(samples, key=lambda s: s.sample_id):
        name = "validation" if sample.text in val else "test" if sample.text in test else "train"
        subsets[name].append(sample.sample_id)
    return {"version": 1, "seed": seed, "group_by": "plate_text",
            "ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
            "labels": {s.sample_id: s.text for s in sorted(samples, key=lambda s: s.sample_id)},
            "splits": subsets}



def write_json(path, data):
    """
        Ghi JSON UTF-8 và tự tạo thư mục cha.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def file_hash(path):
    """
        Mã SHA256 để phát hiện crop hoặc manifest thay đổi sau khi duyệt/train.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_splits(root):
    """
        Đọc bộ chia đã chốt; kiểm tra đủ mẫu, nhãn không đổi và không trùng biển.
    """
    root = Path(root)
    samples = load_dataset(root)
    manifest = json.loads((root/"splits.json").read_text(encoding="utf-8"))
    labels = {s.sample_id: s.text for s in samples}
    
    if manifest.get("labels") != labels:
        raise ValueError("Nhãn/mẫu nguồn đã thay đổi; kiểm tra lại data/splits.json")
    subsets = manifest.get("splits", {})
    if set(subsets) != {"train", "validation", "test"}:
        raise ValueError("Cần đủ ba tập train/validation/test")
    owners, seen = {}, set()
    
    for name, ids in subsets.items():
        for sample_id in ids:
            if sample_id not in labels or sample_id in seen:
                raise ValueError(f"Mẫu lạ hoặc trùng giữa các tập: {sample_id}")
            seen.add(sample_id)
            text = labels[sample_id]
            if owners.setdefault(text, name) != name:
                raise ValueError(f"Biển {text} xuất hiện trong nhiều tập")
    if seen != set(labels):
        raise ValueError("Bộ chia thiếu mẫu nguồn")
    return manifest


def split_samples(root, split):
    """
        Lấy ảnh thuộc một tập cố định, dùng cho đánh giá cả ảnh.
    """
    manifest = load_splits(root)
    
    if split not in manifest["splits"]:
        raise ValueError("split phải là train, validation hoặc test")
    ids = set(manifest["splits"][split])
    return [s for s in load_dataset(root) if s.sample_id in ids]


MANIFEST_FIELDS = ["crop_path", "sample_id", "label", "line", "index", "split", "verified", "sha256"]


def write_character_manifest(path, rows):
    """
        Ghi bảng nguồn gốc crop, nhãn và trạng thái đã duyệt.
    """
    with Path(path).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_character_manifest(root):
    """
        Đọc manifest CSV; file phải có đầy đủ các cột truy vết nguồn.
    """
    path = Path(root)/"characters/manifest.csv"
    
    if not path.exists():
        raise FileNotFoundError("Chưa có ký tự. Chạy python -m src prepare data trước.")
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not set(MANIFEST_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Manifest ký tự thiếu cột bắt buộc")
        return list(reader)


def save_sample_characters(root, sample, split, segmentation):
    """
        Đề xuất nhãn theo thứ tự đọc; mọi crop mới đều verified=false chờ duyệt.
    """
    rows, indexes = [], Counter()
    for char, label in zip(segmentation.characters, sample.text):
        index = indexes[char.line]
        indexes[char.line] += 1
        relative = Path("characters")/split/label/f"{sample.sample_id}_{char.line}_{index}.png"
        path = Path(root)/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), char.image):
            raise OSError(f"Không ghi được crop: {path}")
        rows.append({"crop_path": relative.as_posix(), "sample_id": sample.sample_id,
                     "label": label, "line": char.line, "index": index, "split": split,
                     "verified": "false", "sha256": file_hash(path)})
    return rows


def review_tile(sample, plate, characters):
    """
        Tạo ô đối chiếu: biển gốc bên trái, crop và nhãn đề xuất bên phải.
    """
    tile = np.full((170, 650, 3), 245, np.uint8)
    cv2.putText(tile, f"{sample.sample_id}  {sample.text}", (8, 20), 0, .55, (0, 0, 0), 1)
    ratio = min(165/plate.shape[1], 130/plate.shape[0])
    preview = cv2.resize(plate, (round(plate.shape[1]*ratio), round(plate.shape[0]*ratio)))
    tile[32:32+preview.shape[0], 5:5+preview.shape[1]] = preview
    for i, (char, label) in enumerate(zip(characters, sample.text)):
        x = 180+i*48
        tile[50:122, x:x+42] = cv2.cvtColor(cv2.resize(char.image, (42, 72)), cv2.COLOR_GRAY2BGR)
        cv2.putText(tile, label, (x+10, 43), 0, .65, (180, 0, 0), 2)
        cv2.putText(tile, f"L{char.line}", (x+5, 143), 0, .4, (0, 0, 0), 1)
    return tile


def save_review_pages(directory, tiles):
    """
        Ghép mỗi 8 mẫu vào một trang ảnh để duyệt nhanh, không thay đổi crop.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(tiles), 8):
        page = np.full((680, 1300, 3), 235, np.uint8)
        for index, tile in enumerate(tiles[start:start+8]):
            x, y = index%2*650, index//2*170
            page[y:y+170, x:x+650] = tile
        if not cv2.imwrite(str(directory/f"page_{start//8+1:02d}.jpg"), page):
            raise OSError("Không ghi được bảng ảnh duyệt")


def prepare_characters(root):
    """
        Tách crop đề xuất từ data; bỏ mẫu thiếu/thừa ký tự và xuất ảnh để duyệt.

        Không tự đánh dấu verified chỉ vì số ký tự khớp. Không ghi đè bộ đã chuẩn bị.
    """
    root = Path(root)
    path = root/"characters/manifest.csv"
    if path.exists():
        raise FileExistsError("Bộ crop đã tồn tại; dùng review/train, không tự ghi đè dữ liệu đã duyệt")
    manifest = load_splits(root)
    owners = {sample_id: name for name, ids in manifest["splits"].items() for sample_id in ids}
    rows, proposals, skipped = [], [], []
    pages = {name: [] for name in manifest["splits"]}
    
    for sample in load_dataset(root):
        detection = detect_plate(cv2.imread(str(sample.image_path)))
        if detection.plate is None:
            skipped.append({"sample_id": sample.sample_id, "reason": "no_plate"})
            continue
        segmentation = detection.segmentation or segment_chars(detection.plate)
        if len(segmentation.characters) != len(sample.text):
            skipped.append({"sample_id": sample.sample_id, "reason": "character_count_mismatch"})
            continue
        split = owners[sample.sample_id]
        rows.extend(save_sample_characters(root, sample, split, segmentation))
        pages[split].append(review_tile(sample, detection.plate, segmentation.characters))
        proposals.append({"sample_id": sample.sample_id, "split": split,
                          "page": f"{split}/page_{(len(pages[split])-1)//8+1:02d}.jpg"})
    path.parent.mkdir(parents=True, exist_ok=True)
    write_character_manifest(path, rows)
    for name, tiles in pages.items():
        save_review_pages(root/"characters/review"/name, tiles)
    report = {"proposed_plates": len(proposals), "proposed_characters": len(rows),
              "proposals": proposals, "skipped": skipped,
              "note": "Crop đề xuất chưa được duyệt; verified=false. Không tự coi là ground truth."}
    write_json(root/"characters/preparation.json", report)
    return report


def review_samples(root, sample_ids, accept):
    """
        Đánh dấu chấp nhận/từ chối TOÀN BỘ ký tự của các biển đã xem trên bảng ảnh.
    """
    rows = read_character_manifest(root)
    wanted = set(sample_ids)
    known = {row["sample_id"] for row in rows}
    
    if not wanted or not wanted.issubset(known):
        raise ValueError("Danh sách duyệt rỗng hoặc chứa mẫu không có crop")
    for row in rows:
        if row["sample_id"] in wanted:
            if file_hash(Path(root)/row["crop_path"]) != row["sha256"]:
                raise ValueError("Crop đã thay đổi; cần chuẩn bị/đối chiếu lại trước khi duyệt")
            row["verified"] = "true" if accept else "false"
    write_character_manifest(Path(root)/"characters/manifest.csv", rows)
    return {"reviewed_plates": len(wanted), "accepted": accept}


def load_character_dataset(root, split="train"):
    """
        Chỉ đọc crop đã duyệt, kiểm tra nguồn/tập/nhãn/hash để tránh rò rỉ dữ liệu.
    """
    root = Path(root)
    manifest = load_splits(root)
    if split not in manifest["splits"]:
        raise ValueError("split không hợp lệ")
    allowed = set(manifest["splits"][split])
    images, labels, seen = [], [], set()
    
    for row in read_character_manifest(root):
        if row["split"] != split or row["verified"].lower() != "true":
            continue
        if row["sample_id"] not in allowed:
            raise ValueError("Crop nằm sai tập so với ảnh nguồn")
        path = (root/row["crop_path"]).resolve()
        if not path.is_relative_to((root/"characters").resolve()) or path in seen:
            raise ValueError("Đường dẫn crop không hợp lệ hoặc bị lặp")
        seen.add(path)
        if file_hash(path) != row["sha256"]:
            raise ValueError(f"Crop đã thay đổi sau khi duyệt: {path.name}")
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (48, 32) or not np.isin(image, [0, 255]).all():
            raise ValueError(f"Crop phải là ảnh nhị phân 48x32: {path.name}")
        if re.fullmatch("[A-Z0-9]", row["label"]) is None:
            raise ValueError("Nhãn crop phải là một ký tự chữ hoa/số")
        images.append(image)
        labels.append(row["label"])
    if not images:
        raise ValueError(f"Không có crop đã duyệt trong tập {split}")
    return images, labels


def edit_distance(expected, predicted):
    """
        Levenshtein: số thao tác thêm/xóa/thay ký tự ít nhất để khớp hai chuỗi.
    """
    previous = list(range(len(predicted)+1))
    for i, left in enumerate(expected, 1):
        current = [i]
        for j, right in enumerate(predicted, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(left != right)))
        previous = current
    return previous[-1]


def evaluate(samples, pipeline):
    """
        Đánh giá trên TẤT CẢ ảnh trong tập: lỗi detector/tách chữ cũng được tính.

        Trả độ đúng cả biển, lỗi ký tự và thống kê trạng thái; CER có thể lớn hơn 1.
    """
    if not samples:
        raise ValueError("Tập đánh giá rỗng")
    rows = []
    
    for sample in samples:
        result = pipeline.process(cv2.imread(str(sample.image_path)))
        predicted = result.recognition.text if result.recognition else ""
        rows.append({"sample_id": sample.sample_id, "expected": sample.text, "predicted": predicted,
                     "status": result.status, "exact": predicted == sample.text,
                     "edit_distance": edit_distance(sample.text, predicted),
                     "elapsed_ms": result.elapsed_ms})
    has_ocr = pipeline.recognizer is not None
    return {"count": len(rows), "plate_accuracy": sum(r["exact"] for r in rows)/len(rows) if has_ocr else None,
            "character_error_rate": sum(r["edit_distance"] for r in rows)/sum(len(s.text) for s in samples) if has_ocr else None,
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "mean_elapsed_ms": sum(r["elapsed_ms"] for r in rows)/len(rows), "rows": rows}


def evaluate_characters(recognizer, images, labels):
    """
        Đo OCR riêng trên crop đã duyệt; không nhầm với độ chính xác toàn pipeline.
    """
    predictions = recognizer.predict(images)
    counts, correct = Counter(labels), Counter()
    for expected, actual in zip(labels, predictions):
        correct[expected] += expected == actual
    return {"count": len(labels), "accuracy": sum(correct.values())/len(labels),
            "per_class": {key: {"count": counts[key], "correct": correct[key]} for key in sorted(counts)},
            "unseen_classes": sorted(set(labels)-set(recognizer.metadata["classes"]))}
