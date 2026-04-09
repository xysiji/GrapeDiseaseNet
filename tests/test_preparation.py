from pathlib import Path

from grape_disease_net.data.preparation import (
    collect_detection_records,
    compute_split_counts,
    read_yolo_label_counts,
    stratified_split_records,
)


def test_read_yolo_label_counts(tmp_path: Path) -> None:
    label_path = tmp_path / "sample.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.4 0.1 0.1\n", encoding="utf-8")

    counts = read_yolo_label_counts(label_path)

    assert counts[0] == 1
    assert counts[1] == 1


def test_compute_split_counts_preserves_train_samples() -> None:
    train_count, val_count, test_count = compute_split_counts(5, 0.1, 0.1)
    assert train_count >= 1
    assert train_count + val_count + test_count == 5


def test_collect_and_split_detection_records(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    names = [
        "img_a.rf.1.jpg",
        "img_a.rf.2.jpg",
        "img_b.rf.1.jpg",
        "img_b.rf.2.jpg",
        "img_c.rf.1.jpg",
        "img_d.rf.1.jpg",
    ]
    for index, name in enumerate(names):
        image_stem = Path(name).stem
        (images_dir / name).write_bytes(b"fake-image")
        class_id = 0 if index < 3 else 1
        (labels_dir / f"{image_stem}.txt").write_text(
            f"{class_id} 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

    records = collect_detection_records(images_dir, labels_dir)
    split_map = stratified_split_records(
        records=records,
        class_names=["a", "b"],
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
    )

    assert sum(len(items) for items in split_map.values()) == 6
    assert len(split_map["train"]) >= 2
    source_owners: dict[str, str] = {}
    for split_name, split_records in split_map.items():
        for record in split_records:
            owner = source_owners.get(record.source_key)
            assert owner in (None, split_name)
            source_owners[record.source_key] = split_name
