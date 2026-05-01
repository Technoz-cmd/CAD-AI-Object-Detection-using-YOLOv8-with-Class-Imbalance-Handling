from pathlib import Path
from PIL import Image
import hashlib
from collections import Counter, defaultdict
import os
import re
import cv2
import matplotlib.pyplot as plt
import shutil

# =========================
# CONFIG
# =========================
DATASET_ROOT = Path("dataset")
SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

NUM_CLASSES = 3  # FINAL FIX

# Fix options
FIX_BBOX = True
REMOVE_INVALID_LINES = True
MAKE_BACKUP = True

# Remove duplicated samples created during balancing
REMOVE_BALANCED_COPIES = True
BALANCE_SUFFIX_TAG = "bal"
BALANCE_COPY_PATTERN = re.compile(rf".+_{BALANCE_SUFFIX_TAG}\d{{4}}$")

# =========================
# REPORTS FOLDER
# =========================
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

BACKUP_DIR = REPORTS_DIR / "label_backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# HELPER FUNCTIONS
# =========================
def get_files(folder, exts=None):
    if not folder.exists():
        return []
    if exts is None:
        return list(folder.iterdir())
    return sorted([f for f in folder.iterdir() if f.suffix.lower() in exts])


def file_md5(path):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_image_for_label(img_dir, lbl_stem):
    for ext in IMAGE_EXTS:
        candidate = img_dir / (lbl_stem + ext)
        if candidate.exists():
            return candidate
    return None


def backup_label_file(lbl_path, split):
    if not MAKE_BACKUP:
        return
    split_backup_dir = BACKUP_DIR / split
    split_backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = split_backup_dir / lbl_path.name
    shutil.copy2(lbl_path, backup_path)


def is_balanced_copy_stem(stem: str) -> bool:
    return bool(BALANCE_COPY_PATTERN.fullmatch(stem))


def parse_label_classes(lbl_path):
    classes = []
    try:
        lines = lbl_path.read_text().strip().splitlines()
    except Exception:
        return classes

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
        except Exception:
            continue
        if 0 <= cls < NUM_CLASSES:
            classes.append(cls)
    return classes


def delete_if_exists(path):
    if path and path.exists():
        path.unlink()
        return True
    return False


# =========================
# 0. REMOVE DUPLICATE BALANCED COPIES
# =========================
def remove_balanced_copies():
    print("\n=== REMOVING BALANCED DUPLICATE COPIES ===")
    split = "train"
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"

    removed_images = 0
    removed_labels = 0
    removed_stems = []

    # Remove balanced label files and their matching images
    for lbl_path in list(get_files(lbl_dir, {".txt"})):
        if not is_balanced_copy_stem(lbl_path.stem):
            continue

        img_path = find_image_for_label(img_dir, lbl_path.stem)

        if delete_if_exists(lbl_path):
            removed_labels += 1
        if delete_if_exists(img_path):
            removed_images += 1

        removed_stems.append(lbl_path.stem)

    # Safety pass: remove any balanced images that somehow lost their label
    for img_path in list(get_files(img_dir, IMAGE_EXTS)):
        if is_balanced_copy_stem(img_path.stem):
            if delete_if_exists(img_path):
                removed_images += 1

    report_path = REPORTS_DIR / "remove_balanced_copies_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== REMOVE BALANCED COPIES REPORT ===\n")
        f.write(f"Removed labels: {removed_labels}\n")
        f.write(f"Removed images: {removed_images}\n")
        f.write(f"Removed stems: {len(removed_stems)}\n")
        for stem in removed_stems[:50]:
            f.write(f"{stem}\n")

    print(f"Removed labels: {removed_labels}")
    print(f"Removed images: {removed_images}")
    print(f"Report saved to: {report_path}")


# =========================
# 1. STRUCTURE CHECK
# =========================
def check_structure():
    print("\n=== FOLDER STRUCTURE CHECK ===")
    for split in SPLITS:
        img_dir = DATASET_ROOT / split / "images"
        lbl_dir = DATASET_ROOT / split / "labels"

        print(f"\nSplit: {split}")
        print(f"Images folder exists: {img_dir.exists()}")
        print(f"Labels folder exists: {lbl_dir.exists()}")


# =========================
# 2. IMAGE-LABEL PAIR CHECK
# =========================
def check_pairs(split):
    print(f"\n=== IMAGE-LABEL PAIR CHECK: {split} ===")
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"

    images = [f for f in get_files(img_dir, IMAGE_EXTS)]
    labels = [f for f in get_files(lbl_dir, {".txt"})]

    image_stems = {f.stem for f in images}
    label_stems = {f.stem for f in labels}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    print(f"Total images: {len(images)}")
    print(f"Total labels: {len(labels)}")
    print(f"Images without labels: {len(missing_labels)}")
    print(f"Labels without images: {len(missing_images)}")

    if missing_labels[:10]:
        print("Sample missing labels:", missing_labels[:10])
    if missing_images[:10]:
        print("Sample missing images:", missing_images[:10])

    return missing_labels, missing_images


# =========================
# 3. CORRUPTED IMAGE CHECK
# =========================
def check_corrupted_images(split):
    print(f"\n=== CORRUPTED IMAGE CHECK: {split} ===")
    img_dir = DATASET_ROOT / split / "images"
    bad = []

    for img_path in get_files(img_dir, IMAGE_EXTS):
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            bad.append(img_path.name)

    print(f"Corrupted images: {len(bad)}")
    return bad


# =========================
# 4. EMPTY LABEL CHECK
# =========================
def check_empty_labels(split):
    print(f"\n=== EMPTY LABEL CHECK: {split} ===")
    lbl_dir = DATASET_ROOT / split / "labels"
    empty = []

    for lbl_path in get_files(lbl_dir, {".txt"}):
        if lbl_path.stat().st_size == 0:
            empty.append(lbl_path.name)

    print(f"Empty label files: {len(empty)}")
    return empty


# =========================
# 5. LABEL FORMAT + BBOX VALIDATION
# =========================
def check_label_file(lbl_path, img_path=None):
    issues = []

    try:
        text = lbl_path.read_text().strip()
    except Exception:
        return ["Cannot read label file"]

    if not text:
        return ["Empty label file"]

    lines = text.splitlines()

    img_w, img_h = None, None
    if img_path and img_path.exists():
        with Image.open(img_path) as img:
            img_w, img_h = img.size

    for i, line in enumerate(lines, start=1):
        parts = line.split()

        if len(parts) != 5:
            issues.append(f"Line {i}: wrong format")
            continue

        try:
            cls, x, y, w, h = map(float, parts)
        except Exception:
            issues.append(f"Line {i}: non-numeric values")
            continue

        if int(cls) < 0 or int(cls) >= NUM_CLASSES:
            issues.append(f"Line {i}: invalid class id")

        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            issues.append(f"Line {i}: not normalized")

        if img_w and img_h:
            x1 = (x - w / 2) * img_w
            y1 = (y - h / 2) * img_h
            x2 = (x + w / 2) * img_w
            y2 = (y + h / 2) * img_h

            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                issues.append(f"Line {i}: box outside image")

    return issues


def check_labels(split):
    print(f"\n=== LABEL FORMAT CHECK: {split} ===")
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"

    bad_files = {}

    for lbl_path in get_files(lbl_dir, {".txt"}):
        img_path = find_image_for_label(img_dir, lbl_path.stem)
        issues = check_label_file(lbl_path, img_path)
        if issues:
            bad_files[lbl_path.name] = issues

    print(f"Label files with issues: {len(bad_files)}")

    report_path = REPORTS_DIR / f"bad_labels_{split}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        for fname, issues in bad_files.items():
            f.write(f"{fname}\n")
            for issue in issues:
                f.write(f"  - {issue}\n")

    return bad_files


# =========================
# 5B. FIX LABELS
# =========================
def fix_label_file(lbl_path, img_path):
    try:
        text = lbl_path.read_text().strip()
    except Exception:
        return False, 0, 0

    if not text:
        return False, 0, 0

    with Image.open(img_path) as img:
        img_w, img_h = img.size

    original_lines = text.splitlines()
    fixed_lines = []
    changed = 0
    skipped = 0

    for line in original_lines:
        parts = line.split()

        if len(parts) != 5:
            skipped += 1
            continue

        try:
            cls, x, y, w, h = map(float, parts)
        except Exception:
            skipped += 1
            continue

        cls_id = int(cls)
        if cls_id < 0 or cls_id >= NUM_CLASSES:
            skipped += 1
            continue

        # Convert YOLO -> pixel coords
        x1 = (x - w / 2) * img_w
        y1 = (y - h / 2) * img_h
        x2 = (x + w / 2) * img_w
        y2 = (y + h / 2) * img_h

        # Clip boxes to image boundaries
        if FIX_BBOX:
            new_x1 = max(0.0, x1)
            new_y1 = max(0.0, y1)
            new_x2 = min(float(img_w), x2)
            new_y2 = min(float(img_h), y2)
        else:
            new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2

        if REMOVE_INVALID_LINES and (new_x2 <= new_x1 or new_y2 <= new_y1):
            skipped += 1
            continue

        # Convert back to YOLO format
        new_x = ((new_x1 + new_x2) / 2.0) / img_w
        new_y = ((new_y1 + new_y2) / 2.0) / img_h
        new_w = (new_x2 - new_x1) / img_w
        new_h = (new_y2 - new_y1) / img_h

        # Final clamp
        new_x = min(max(new_x, 0.0), 1.0)
        new_y = min(max(new_y, 0.0), 1.0)
        new_w = min(max(new_w, 0.0), 1.0)
        new_h = min(max(new_h, 0.0), 1.0)

        if new_w <= 0 or new_h <= 0:
            skipped += 1
            continue

        fixed_line = f"{cls_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}"
        fixed_lines.append(fixed_line)

        if fixed_line != line:
            changed += 1

    # Write back cleaned label file
    if fixed_lines:
        lbl_path.write_text("\n".join(fixed_lines) + "\n", encoding="utf-8")
        return True, changed, skipped
    else:
        # Keep empty file rather than deleting, so image-label pairing stays intact
        lbl_path.write_text("", encoding="utf-8")
        return False, changed, skipped


def fix_labels(split):
    print(f"\n=== FIXING LABELS: {split} ===")
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"

    total_files = 0
    modified_files = 0
    total_changed_lines = 0
    total_skipped_lines = 0

    for lbl_path in get_files(lbl_dir, {".txt"}):
        img_path = find_image_for_label(img_dir, lbl_path.stem)
        if img_path is None:
            continue

        total_files += 1
        backup_label_file(lbl_path, split)

        success, changed, skipped = fix_label_file(lbl_path, img_path)
        if changed > 0 or skipped > 0:
            modified_files += 1
        total_changed_lines += changed
        total_skipped_lines += skipped

    print(f"Label files processed: {total_files}")
    print(f"Label files modified: {modified_files}")
    print(f"Changed lines: {total_changed_lines}")
    print(f"Skipped/removed lines: {total_skipped_lines}")

    report_path = REPORTS_DIR / f"fix_report_{split}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Processed files: {total_files}\n")
        f.write(f"Modified files: {modified_files}\n")
        f.write(f"Changed lines: {total_changed_lines}\n")
        f.write(f"Skipped/removed lines: {total_skipped_lines}\n")

    return {
        "processed": total_files,
        "modified": modified_files,
        "changed_lines": total_changed_lines,
        "skipped_lines": total_skipped_lines,
    }


# =========================
# 6. CLASS DISTRIBUTION
# =========================
def class_distribution(split):
    print(f"\n=== CLASS DISTRIBUTION: {split} ===")
    lbl_dir = DATASET_ROOT / split / "labels"
    counter = Counter()

    for lbl_path in get_files(lbl_dir, {".txt"}):
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls = int(float(parts[0]))
            except Exception:
                continue
            if cls < 0 or cls >= NUM_CLASSES:
                continue
            counter[cls] += 1

    report_path = REPORTS_DIR / f"class_distribution_{split}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        for k, v in sorted(counter.items()):
            f.write(f"Class {k}: {v}\n")

    for k, v in sorted(counter.items()):
        print(f"Class {k}: {v}")

    return counter


# =========================
# 7. DUPLICATE IMAGE CHECK
# =========================
def check_duplicates(split):
    print(f"\n=== DUPLICATE IMAGE CHECK: {split} ===")
    img_dir = DATASET_ROOT / split / "images"
    hashes = defaultdict(list)

    for img_path in get_files(img_dir, IMAGE_EXTS):
        try:
            hashes[file_md5(img_path)].append(img_path.name)
        except Exception:
            pass

    duplicates = {k: v for k, v in hashes.items() if len(v) > 1}

    print(f"Duplicate groups found: {len(duplicates)}")
    return duplicates


# =========================
# 8. VISUAL INSPECTION (SAVE IMAGES)
# =========================
def show_samples(split, num_samples=5):
    print(f"\n=== SAVING VISUAL SAMPLES: {split} ===")

    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"

    images = get_files(img_dir, IMAGE_EXTS)[:num_samples]

    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue

                try:
                    cls, x, y, bw, bh = map(float, parts)
                except Exception:
                    continue

                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        save_path = REPORTS_DIR / f"{split}_{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), img)


# =========================
# MAIN RUNNER
# =========================
def run_all():
    check_structure()

    if REMOVE_BALANCED_COPIES:
        remove_balanced_copies()

    print("\n=== PRE-FIX DATASET CHECKS ===")
    for split in SPLITS:
        check_pairs(split)
        check_corrupted_images(split)
        check_empty_labels(split)
        check_labels(split)
        class_distribution(split)
        check_duplicates(split)

    print("\n=== APPLYING LABEL FIXES ===")
    for split in SPLITS:
        fix_labels(split)

    print("\n=== POST-FIX DATASET CHECKS ===")
    for split in SPLITS:
        check_labels(split)
        class_distribution(split)

    show_samples("train", num_samples=5)

    print("\n✅ PREPROCESSING COMPLETE")
    print("👉 Check 'reports/' folder for outputs")
    print("👉 Original labels were backed up in 'reports/label_backups/'")


if __name__ == "__main__":
    run_all()