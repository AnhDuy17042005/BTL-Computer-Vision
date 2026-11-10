"""
    Giao diện chọn ảnh, nhận dạng biển số, xem từng bước và train SVM.
"""

import sys
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QProcess
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QGridLayout, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget)

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
FROZEN = getattr(sys, "frozen", False)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ocr import SVMRecognizer
from src.pipeline import PlatePipeline, read_image, save_result

STATUS_TEXT = {
    "ok": "Đã nhận dạng thành công",
    "no_plate": "Chưa tìm được biển số",
    "no_characters": "Chưa tách được ký tự",
    "no_model": "Chưa nạp model; hãy chọn model hoặc train SVM",
    "invalid_text": "Chuỗi nhận dạng chưa đúng định dạng biển số",
}


class ImageLabel(QLabel):
    """
        Hiển thị ảnh OpenCV, giữ tỷ lệ khi thay đổi kích thước cửa sổ.
    """

    def __init__(self, text="Chưa có ảnh"):
        """
            Tạo vùng xem ảnh với hướng dẫn ban đầu.
        """
        super().__init__(text)
        self.source = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, 180)
        self.setStyleSheet("background: #edf1f5; color: #334155; border: 1px solid #cbd5e1;")

    def set_image(self, image):
        """
            Chuyển ảnh xám hoặc BGR thành QPixmap, sao chép để giữ bộ nhớ ảnh.
        """
        self.source = None
        if image is None:
            self.clear()
            self.setText("Chưa có ảnh")
            return
        conversion = cv2.COLOR_GRAY2RGB if image.ndim == 2 else cv2.COLOR_BGR2RGB
        rgb = cv2.cvtColor(image, conversion)
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
        self.source = QPixmap.fromImage(qimage.copy())
        self.update_image()

    def update_image(self):
        """
            Co ảnh vừa khung mà không kéo méo ảnh.
        """
        if self.source is not None:
            self.setPixmap(self.source.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        """
            Cập nhật ảnh khi vùng hiển thị được thay đổi kích thước.
        """
        super().resizeEvent(event)
        self.update_image()


class MainWindow(QMainWindow):
    """
        Quản lý ảnh đang mở, model OCR, kết quả và tiến trình train.
    """

    def __init__(self):
        """
            Tạo giao diện và tự nạp model mặc định nếu file đã tồn tại.
        """
        super().__init__()
        self.image = None
        self.image_path = None
        self.result = None
        self.recognizer = None
        self.train_process = None
        self.setWindowTitle("Nhận dạng biển số xe")
        self.resize(1200, 900)
        self.setMinimumSize(900, 650)
        self.build_ui()
        model_path = ROOT/"models/ocr_svm.xml"
        if model_path.is_file():
            self.load_model(model_path)
        self.statusBar().showMessage("Chọn ảnh để bắt đầu nhận dạng")

    def build_ui(self):
        """
            Tạo thanh nút thao tác và ba tab kết quả, ảnh trung gian, train.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        toolbar = QHBoxLayout()
        actions = [("Chọn ảnh", self.choose_image), ("Chọn model", self.choose_model),
                   ("Nhận dạng", self.run_recognition), ("Lưu kết quả", self.export_result),
                   ("Train SVM", self.start_training)]
        self.buttons = {}
        for title, handler in actions:
            button = QPushButton(title)
            button.clicked.connect(handler)
            toolbar.addWidget(button)
            self.buttons[title] = button
        layout.addLayout(toolbar)
        self.model_label = QLabel("Chưa có model. Chọn model đã train hoặc bấm Train SVM.")
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)
        self.tabs = QTabWidget()
        self.build_result_tab()
        self.steps_scroll = QScrollArea()
        self.steps_scroll.setWidgetResizable(True)
        self.tabs.addTab(self.steps_scroll, "Ảnh trung gian")
        self.show_steps()
        self.training_log = QPlainTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setPlainText("Bấm Train SVM để học từ bộ ký tự đã duyệt trong data/.\n"
                                       "Model được lưu tại models/ocr_svm.xml và tự nạp khi train xong.")
        self.tabs.addTab(self.training_log, "Train SVM")
        if FROZEN:
            self.buttons["Train SVM"].setEnabled(False)
            self.training_log.setPlainText("Bản đóng gói dùng model đã train. "
                                           "Để train lại, chạy `pixi run train` trong thư mục project.")
        layout.addWidget(self.tabs)
        self.setCentralWidget(container)
        self.buttons["Nhận dạng"].setEnabled(False)
        self.buttons["Lưu kết quả"].setEnabled(False)

    def build_result_tab(self):
        """
            Tạo vùng ảnh đầu vào, ảnh biển cắt được và thông tin nhận dạng.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.file_label = QLabel("Chưa chọn ảnh")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        images = QHBoxLayout()
        self.original_label = ImageLabel("Chọn ảnh xe bằng nút Chọn ảnh")
        self.plate_label = ImageLabel("Ảnh biển số sẽ xuất hiện sau khi nhận dạng")
        images.addWidget(self.original_label, 2)
        images.addWidget(self.plate_label, 1)
        layout.addLayout(images, 1)
        self.plate_text = QLabel("Biển số: —")
        self.plate_text.setStyleSheet("font-size: 28px; font-weight: bold;")
        self.details_label = QLabel("Chọn ảnh rồi bấm Nhận dạng.")
        self.details_label.setWordWrap(True)
        self.details_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.plate_text)
        layout.addWidget(self.details_label)
        self.tabs.addTab(tab, "Kết quả")

    def choose_image(self):
        """
            Mở hộp thoại chọn ảnh và báo lỗi nếu file không đọc được.
        """
        folder = Path.home() if FROZEN else ROOT/"data/images"
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", str(folder),
                                              "Ảnh (*.jpg *.jpeg *.png *.bmp)")
        if path:
            try:
                self.load_image(path)
            except (OSError, ValueError, cv2.error) as exc:
                QMessageBox.warning(self, "Không mở được ảnh", str(exc))

    def load_image(self, path):
        """
            Nạp ảnh mới và xóa kết quả cũ để tránh hiển thị nhầm biển số.
        """
        image = read_image(path)
        self.image, self.image_path, self.result = image, Path(path), None
        self.file_label.setText(f"Ảnh: {self.image_path.name}")
        self.original_label.set_image(image)
        self.plate_label.set_image(None)
        self.plate_text.setText("Biển số: —")
        self.details_label.setText("Bấm Nhận dạng để xử lý ảnh.")
        self.show_steps()
        self.buttons["Nhận dạng"].setEnabled(True)
        self.buttons["Lưu kết quả"].setEnabled(False)
        self.tabs.setCurrentIndex(0)

    def choose_model(self):
        """
            Chọn model XML có file metadata .xml.json nằm cạnh model.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Chọn model", str(ROOT/"models"), "SVM (*.xml)")
        if path:
            self.load_model(path)

    def load_model(self, path):
        """
            Nạp model OCR; nếu lỗi thì giữ model đang dùng và hiển thị lý do.
        """
        try:
            recognizer = SVMRecognizer.load(path)
        except (OSError, ValueError, cv2.error) as exc:
            self.model_label.setText(f"Không nạp được model: {exc}. Model đang dùng được giữ nguyên.")
            return False
        self.recognizer = recognizer
        self.model_label.setText(f"Model: {path} — {len(recognizer.metadata['classes'])} lớp ký tự")
        return True

    def run_recognition(self):
        """
            Chạy pipeline cho ảnh đang mở và cập nhật kết quả lên giao diện.
        """
        if self.image is None:
            return
        try:
            result = PlatePipeline(self.recognizer).process(self.image)
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            QMessageBox.warning(self, "Không xử lý được ảnh", str(exc))
            return
        
        self.result = result
        self.original_label.set_image(result.debug.get("detection", self.image))
        self.plate_label.set_image(result.detection.plate)
        text = result.recognition.text if result.recognition else "—"
        self.plate_text.setText(f"Biển số: {text}")
        self.details_label.setText(f"{STATUS_TEXT[result.status]}\n"
                                   f"Ký tự: {len(result.segmentation.characters)} | "
                                   f"Thời gian: {result.elapsed_ms:.1f} ms | Bbox: {result.detection.bbox}")
        self.show_steps()
        self.buttons["Lưu kết quả"].setEnabled(True)
        self.tabs.setCurrentIndex(0)

    def show_steps(self):
        """
            Hiển thị ảnh trung gian và từng ký tự; giải phóng vùng xem cũ.
        """
        old = self.steps_scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        container = QWidget()
        layout = QGridLayout(container)
        images = dict(self.result.debug) if self.result is not None else {}
        if self.result is not None:
            images.update({f"Ký tự {index+1}": char.image
                           for index, char in enumerate(self.result.segmentation.characters)})
        for index, (name, image) in enumerate(images.items()):
            tile = QWidget()
            tile_layout = QVBoxLayout(tile)
            tile_layout.addWidget(QLabel(name))
            preview = ImageLabel()
            preview.set_image(image)
            tile_layout.addWidget(preview)
            layout.addWidget(tile, index//3, index%3)
        if not images:
            layout.addWidget(QLabel("Chạy nhận dạng để xem các bước xử lý ảnh."), 0, 0)
        self.steps_scroll.setWidget(container)

    def export_result(self):
        """
            Lưu JSON, CSV và ảnh xử lý vào thư mục theo tên ảnh đầu vào.
        """
        if self.result is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu", str(Path.home()))
        if directory:
            try:
                destination = Path(directory)/f"{self.image_path.stem}_result"
                save_result(self.result, destination)
                self.statusBar().showMessage(f"Đã lưu: {destination}")
            except (OSError, ValueError, cv2.error) as exc:
                QMessageBox.warning(self, "Không lưu được kết quả", str(exc))

    def start_training(self):
        """
            Chạy lệnh train trong tiến trình riêng để giao diện vẫn phản hồi.
        """
        if FROZEN or self.train_process is not None:
            return
        self.train_process = QProcess(self)
        self.train_process.setWorkingDirectory(str(ROOT))
        self.train_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.train_process.readyReadStandardOutput.connect(self.read_training_output)
        self.train_process.finished.connect(self.finish_training)
        self.train_process.errorOccurred.connect(self.training_error)
        self.buttons["Train SVM"].setEnabled(False)
        self.training_log.appendPlainText("\nĐang train SVM từ các crop đã duyệt...")
        self.tabs.setCurrentIndex(2)
        self.train_process.start(sys.executable, ["-u", "-m", "src", "train", "data"])

    def read_training_output(self):
        """
            Đưa log của tiến trình train vào tab Train SVM.
        """
        if self.train_process is not None:
            output = bytes(self.train_process.readAllStandardOutput()).decode("utf-8", errors="replace")
            self.training_log.appendPlainText(output.rstrip())

    def training_error(self, error):
        """
            Bật lại nút khi tiến trình train không khởi động được.
        """
        if error == QProcess.ProcessError.FailedToStart:
            self.training_log.appendPlainText("Không khởi động được tiến trình train.")
            self.finish_training(1, QProcess.ExitStatus.CrashExit)

    def finish_training(self, code, status):
        """
            Giải phóng tiến trình, bật lại nút và nạp model sau khi train thành công.
        """
        self.read_training_output()
        success = code == 0 and status == QProcess.ExitStatus.NormalExit
        self.training_log.appendPlainText("Train hoàn tất." if success else f"Train thất bại, mã lỗi {code}.")
        if success:
            self.load_model(ROOT/"models/ocr_svm.xml")
        if self.train_process is not None:
            self.train_process.deleteLater()
            self.train_process = None
        self.buttons["Train SVM"].setEnabled(True)

    def closeEvent(self, event):
        """
            Chờ train hoàn tất trước khi đóng để không ngắt lúc đang ghi model.
        """
        if self.train_process is not None:
            self.statusBar().showMessage("Vui lòng chờ train hoàn tất trước khi đóng cửa sổ.")
            event.ignore()
        else:
            event.accept()


def main():
    """
        Tạo ứng dụng Qt và hiển thị cửa sổ nhận dạng biển số.
    """
    if len(sys.argv) == 3 and sys.argv[1] == "--check-image":
        model = SVMRecognizer.load(ROOT/"models/ocr_svm.xml")
        result = PlatePipeline(model).process(read_image(sys.argv[2]))
        print(result.recognition.text if result.recognition else result.status)
        return 0 if result.status == "ok" else 1
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
