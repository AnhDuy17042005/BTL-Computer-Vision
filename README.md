# License Plate Recognition with OCR

Chương trình nhận dạng biển số xe từ ảnh hoặc video bằng xử lý ảnh, phát hiện vùng biển số và OCR. Dự án có giao diện GUI cho phép người dùng upload ảnh/video, hiển thị kết quả nhận dạng và trực quan hóa vùng biển số được phát hiện.

## 1. Mục Tiêu Dự Án

Dự án hướng đến xây dựng một hệ thống nhận dạng biển số xe cơ bản gồm các chức năng chính:

- Upload ảnh từ máy tính.
- Upload video từ máy tính.
- Phát hiện vùng biển số xe trong ảnh hoặc từng frame video.
- Tiền xử lý vùng biển số để cải thiện chất lượng OCR.
- Nhận dạng ký tự trên biển số bằng OCR.
- Hiển thị ảnh/video đầu vào, vùng biển số được phát hiện và kết quả nhận dạng trên GUI.

## 2. Công Nghệ Sử Dụng

- Python
- OpenCV
- NumPy
- Pillow
- Matplotlib
- PySide6
- OCR engine: dự kiến sử dụng EasyOCR, PaddleOCR hoặc Tesseract OCR
- Môi trường quản lý bằng Pixi

## 3. Cấu Trúc Dự Án Ban Đầu

```text
BTL/
├── README.md
├── pixi.toml
├── pixi.lock
├── data/
├── outputs/
├── gui/
│   └── main.py
└── src/
    ├── preprocessing.py
    └── ocr.py
