"""
    CLI: audit → prepare → review → train → run/evaluate. Xem python -m src --help.
"""

import argparse
import json
from pathlib import Path
import cv2

from .dataset import (audit_dataset, load_dataset, split_dataset, prepare_characters,
                      review_samples, load_character_dataset, split_samples, evaluate,
                      evaluate_characters, file_hash, write_json)
from .detect import DetectorConfig
from .ocr import SVMRecognizer
from .pipeline import PipelineConfig, PlatePipeline, read_image, result_summary, save_result


def print_json(value):
    """
        In dictionary dễ đọc, giữ tiếng Việt trong terminal.
    """
    print(json.dumps(value, indent=2, ensure_ascii=False))


def command_audit(args):
    """
        Kiểm tra ảnh/nhãn và trả exit code 1 nếu dữ liệu không hợp lệ.
    """
    report = audit_dataset(args.data)
    print_json(report)
    return 0 if report["ok"] else 1


def command_split(args):
    """
        Tạo bộ chia mới; không ghi đè splits mà nhóm đang sử dụng.
    """
    report = split_dataset(load_dataset(args.data), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print_json({name: len(ids) for name, ids in report["splits"].items()})
    return 0


def command_prepare(args):
    """
        Tạo crop đề xuất và bảng ảnh; bước này chưa cấp quyền dùng crop để train.
    """
    report = prepare_characters(args.data)
    print_json({"proposed_plates": report["proposed_plates"],
                "proposed_characters": report["proposed_characters"],
                "skipped": len(report["skipped"]), "review": str(args.data/"characters/review")})
    return 0


def command_review(args):
    """
        Ghi quyết định duyệt sau khi người dùng đã xem ảnh và thứ tự ký tự.
    """
    print_json(review_samples(args.data, args.accept or args.reject, bool(args.accept)))
    return 0


def command_train(args):
    """
        Train bằng crop verified của TRAIN; validation chỉ dùng đo kết quả.
    """
    images, labels = load_character_dataset(args.data, "train")
    recognizer = SVMRecognizer()
    recognizer.train(images, labels, c=args.c, augment=not args.no_augment)
    recognizer.metadata.update({"train_split": "train", "opencv_version": cv2.__version__,
                               "split_sha256": file_hash(args.data/"splits.json"),
                               "character_manifest_sha256": file_hash(args.data/"characters/manifest.csv")})
    recognizer.save(args.model)
    report = {"training": recognizer.metadata,
              "train_characters": evaluate_characters(recognizer, images, labels)}
    try:
        val_images, val_labels = load_character_dataset(args.data, "validation")
    except ValueError as exc:

        report["validation_error"] = str(exc)
    else:
        report["validation_characters"] = evaluate_characters(recognizer, val_images, val_labels)
    write_json(args.report, report)
    print_json({"model": str(args.model), "training_samples": len(images), "classes": len(set(labels)),
                "validation_accuracy": report.get("validation_characters", {}).get("accuracy"),
                "report": str(args.report)})
    return 0


def make_pipeline(args):
    """
        Nạp model và cấu hình detector; cho phép --no-model để kiểm tra xử lý ảnh.
    """
    recognizer = None if args.no_model else SVMRecognizer.load(args.model)
    config = PipelineConfig(detector=DetectorConfig(method=args.detector))
    return PlatePipeline(recognizer, config)


def command_run(args):
    """
        Chạy một ảnh, in kết quả và lưu ảnh trung gian/JSON/CSV.
    """
    result = make_pipeline(args).process(read_image(args.image))
    save_result(result, args.output)
    print_json(result_summary(result))
    return 0


def command_evaluate(args):
    """
        Đánh giá toàn bộ ảnh của tập chọn, không bỏ các ảnh xử lý thất bại.
    """
    pipeline = make_pipeline(args)
    report = evaluate(split_samples(args.data, args.split), pipeline)
    report["split"] = args.split
    report["split_sha256"] = file_hash(args.data/"splits.json")
    if pipeline.recognizer is not None:
        try:
            images, labels = load_character_dataset(args.data, args.split)
        except ValueError as exc:
            report["character_evaluation_error"] = str(exc)
        else:
            report["verified_characters"] = evaluate_characters(pipeline.recognizer, images, labels)
    write_json(args.output, report)
    print_json({key: value for key, value in report.items() if key not in {"rows", "verified_characters"}})
    return 0


def add_pipeline_options(parser):
    """
        Các tham số dùng chung khi chạy nhận dạng hoặc đánh giá.
    """
    parser.add_argument("--model", type=Path, default=Path("models/ocr_svm.xml"))
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--detector", choices=["edge", "bright", "both"], default="both")


def build_parser():
    """
        Khai báo các lệnh; mỗi lệnh được xử lý bằng một hàm nhỏ riêng.
    """
    parser = argparse.ArgumentParser(description="Nhận dạng biển số từ ảnh — HOG + SVM")
    commands = parser.add_subparsers(dest="command", required=True)
    handlers = {"audit": command_audit, "split": command_split, "prepare": command_prepare,
                "review": command_review, "train": command_train, "evaluate": command_evaluate}
    for name, handler in handlers.items():
        command = commands.add_parser(name)
        command.add_argument("data", type=Path, nargs="?", default=Path("data"))
        command.set_defaults(handler=handler)
        if name == "split":
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--output", type=Path, default=Path("data/splits.json"))
        elif name == "review":
            choice = command.add_mutually_exclusive_group(required=True)
            choice.add_argument("--accept", nargs="+", metavar="SAMPLE_ID")
            choice.add_argument("--reject", nargs="+", metavar="SAMPLE_ID")
        elif name == "train":
            command.add_argument("--model", type=Path, default=Path("models/ocr_svm.xml"))
            command.add_argument("--c", type=float, default=2.5)
            command.add_argument("--no-augment", action="store_true")
            command.add_argument("--report", type=Path, default=Path("outputs/training_report.json"))
        elif name == "evaluate":
            add_pipeline_options(command)
            command.add_argument("--split", choices=["train", "validation", "test"], default="test")
            command.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    run = commands.add_parser("run")
    run.add_argument("image", type=Path)
    add_pipeline_options(run)
    run.add_argument("--output", type=Path, default=Path("outputs/demo"))
    run.set_defaults(handler=command_run)
    return parser


def main():
    """
        Điểm vào CLI; lỗi được in gọn và trả exit code 2.
    """
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.handler(args)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.exit(2, f"Lỗi: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
