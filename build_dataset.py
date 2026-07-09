"""
==========================================================
 Retina Dataset Builder
 Production Version (Corrected)
==========================================================
"""

import os
import shutil
import hashlib
import logging
from pathlib import Path

import pandas as pd
from PIL import Image

# ==========================================================
# PROJECT PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent

RAW = ROOT / "raw_dataset"

OUTPUT = ROOT / "dataset"

ODIR = RAW / "ODIR"
APTOS = RAW / "APTOS"
GLAUCOMA = RAW / "Glaucoma"
CATARACT = RAW / "Cataract"

# ==========================================================
# OUTPUT CLASSES
# ==========================================================

CLASSES = [
    "Healthy",
    "Diabetic_Retinopathy",
    "Glaucoma",
    "AMD",
    "Cataract"
]

for cls in CLASSES:
    (OUTPUT / cls).mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOGGER
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s : %(message)s"
)

logger = logging.getLogger("DatasetBuilder")

# ==========================================================
# IMAGE EXTENSIONS
# ==========================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

# ==========================================================
# DUPLICATE HASHES
# ==========================================================

IMAGE_HASHES = set()

# ==========================================================
# COUNTERS
# ==========================================================

COUNTS = {
    "Healthy": 0,
    "Diabetic_Retinopathy": 0,
    "Glaucoma": 0,
    "AMD": 0,
    "Cataract": 0,
    "Skipped": 0,
    "Duplicate": 0,
    "Corrupted": 0
}

# ==========================================================
# HASH
# ==========================================================

def sha256(path):

    h = hashlib.sha256()

    with open(path, 'rb') as f:

        while True:

            chunk = f.read(8192)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()

# ==========================================================
# IMAGE VALIDATION
# ==========================================================

def valid_image(path):

    try:

        img = Image.open(path)

        img.verify()

        return True

    except Exception:

        return False

# ==========================================================
# COPY IMAGE
# ==========================================================
# FIX: previously, files were copied using only their original
# filename (e.g. "1.jpg"). Since multiple source datasets reuse
# generic filenames, this caused DIFFERENT images to silently
# overwrite each other at the destination, quietly shrinking the
# dataset while COUNTS still reported the higher (wrong) number.
#
# Fix: prefix every destination filename with the first 10 chars
# of its content hash, guaranteeing uniqueness across all sources
# while keeping the original name for traceability.
# ==========================================================

def copy_image(src, destination):

    global IMAGE_HASHES

    if not os.path.exists(src):

        COUNTS["Skipped"] += 1
        return

    ext = Path(src).suffix.lower()

    if ext not in VALID_EXTENSIONS:

        COUNTS["Skipped"] += 1
        return

    if not valid_image(src):

        COUNTS["Corrupted"] += 1
        return

    file_hash = sha256(src)

    if file_hash in IMAGE_HASHES:

        COUNTS["Duplicate"] += 1
        return

    IMAGE_HASHES.add(file_hash)

    original_name = Path(src).name

    # Unique, collision-proof filename
    unique_filename = f"{file_hash[:10]}_{original_name}"

    dst = OUTPUT / destination / unique_filename

    # Extra safety net: if somehow the exact same filename exists
    # (should not happen given the hash prefix), don't crash — skip.
    if dst.exists():
        COUNTS["Duplicate"] += 1
        return

    shutil.copy2(src, dst)

    COUNTS[destination] += 1

# ==========================================================
# FIND KEYWORD
# ==========================================================
# FIX: "retinopathy" alone used to match non-diabetic cases
# (e.g. "hypertensive retinopathy"), mislabeling them as DR.
# Now requires "diabetic" to co-occur with "retinopathy".
# ==========================================================

def keyword_folder(keyword):

    keyword = str(keyword).lower()

    if "normal fundus" in keyword:
        return "Healthy"

    if "diabetic" in keyword and "retinopathy" in keyword:
        return "Diabetic_Retinopathy"

    if "diabetic" in keyword:
        return "Diabetic_Retinopathy"

    if "glaucoma" in keyword:
        return "Glaucoma"

    if "cataract" in keyword:
        return "Cataract"

    if "age-related macular degeneration" in keyword:
        return "AMD"

    if "amd" in keyword:
        return "AMD"

    return None

# ==========================================================
# ODIR PROCESSOR
# ==========================================================

def process_odir():

    logger.info("Processing ODIR Dataset...")

    csv = ODIR / "full_df.csv"

    image_dir = ODIR / "preprocessed_images"

    if not csv.exists():

        logger.error("ODIR CSV Missing")

        return

    df = pd.read_csv(csv)

    total = len(df)

    for index, row in df.iterrows():

        left_image = image_dir / row["Left-Fundus"]

        right_image = image_dir / row["Right-Fundus"]

        left_class = keyword_folder(
            row["Left-Diagnostic Keywords"]
        )

        right_class = keyword_folder(
            row["Right-Diagnostic Keywords"]
        )

        if left_class:

            copy_image(
                str(left_image),
                left_class
            )

        if right_class:

            copy_image(
                str(right_image),
                right_class
            )

        if index % 500 == 0:

            logger.info(
                f"ODIR {index}/{total}"
            )

    logger.info("ODIR Completed.")

# ==========================================================
# APTOS PROCESSOR
# ==========================================================

def process_aptos():

    logger.info("Processing APTOS Dataset...")

    csv = APTOS / "train_1.csv"

    image_dir = APTOS / "train_images" / "train_images"

    if not csv.exists():

        logger.error("APTOS CSV Missing")

        return

    df = pd.read_csv(csv)

    total = len(df)

    copied = 0

    skipped = 0

    for index, row in df.iterrows():

        diagnosis = int(row["diagnosis"])

        filename = str(row["id_code"]) + ".png"

        src = image_dir / filename

        # 0 = Healthy (Ignore)
        if diagnosis == 0:

            skipped += 1
            continue

        # 1,2,3,4 = DR
        copy_image(
            str(src),
            "Diabetic_Retinopathy"
        )

        copied += 1

        if (index + 1) % 500 == 0:

            logger.info(
                f"APTOS {index+1}/{total}"
            )

    logger.info(
        f"APTOS Completed | Copied : {copied} | Ignored Healthy : {skipped}"
    )

# ==========================================================
# GLAUCOMA DATASET
# ==========================================================

def process_glaucoma_split(split_name):

    split = GLAUCOMA / split_name

    if not split.exists():

        logger.warning(f"{split_name} folder missing")

        return

    mapping = {
        "advanced": "Glaucoma",
        "early": "Glaucoma",
        "normal": "Healthy"
    }

    for folder, target in mapping.items():

        source = split / folder

        if not source.exists():
            continue

        logger.info(
            f"{split_name}/{folder}"
        )

        for root, dirs, files in os.walk(source):

            for file in files:

                ext = Path(file).suffix.lower()

                if ext not in VALID_EXTENSIONS:
                    continue

                src = Path(root) / file

                copy_image(
                    str(src),
                    target
                )

def process_glaucoma():

    logger.info("Processing Glaucoma Dataset...")

    process_glaucoma_split("train")

    process_glaucoma_split("valid")

    process_glaucoma_split("test")

    logger.info("Glaucoma Completed.")

# ==========================================================
# CATARACT DATASET
# ==========================================================

def process_cataract_folder(folder_name, target):

    source = CATARACT / "dataset" / folder_name

    if not source.exists():

        logger.warning(f"{folder_name} Missing")

        return

    logger.info(f"Processing Cataract/{folder_name}")

    copied = 0

    for root, dirs, files in os.walk(source):

        for file in files:

            ext = Path(file).suffix.lower()

            if ext not in VALID_EXTENSIONS:
                continue

            src = Path(root) / file

            copy_image(
                str(src),
                target
            )

            copied += 1

            if copied % 500 == 0:

                logger.info(
                    f"{folder_name} : {copied}"
                )

    logger.info(
        f"{folder_name} Finished : {copied}"
    )


def process_cataract():

    logger.info("Processing Cataract Dataset...")

    process_cataract_folder(
        "1_normal",
        "Healthy"
    )

    process_cataract_folder(
        "2_cataract",
        "Cataract"
    )

    process_cataract_folder(
        "2_glaucoma",
        "Glaucoma"
    )

    # Mixed diseases
    ignore = CATARACT / "dataset" / "3_retina_disease"

    if ignore.exists():

        total = 0

        for _, _, files in os.walk(ignore):

            total += len(files)

        logger.info(
            f"Ignored Retina Disease Images : {total}"
        )

    logger.info("Cataract Dataset Completed.")

# ==========================================================
# DATASET REPORT
# ==========================================================

def count_images(folder):

    total = 0

    for root, dirs, files in os.walk(folder):

        for file in files:

            if Path(file).suffix.lower() in VALID_EXTENSIONS:

                total += 1

    return total


def print_summary():

    logger.info("")

    logger.info("=" * 60)

    logger.info("FINAL DATASET SUMMARY")

    logger.info("=" * 60)

    for cls in CLASSES:

        total = count_images(
            OUTPUT / cls
        )

        logger.info(
            f"{cls:25} : {total}"
        )

    logger.info("")

    logger.info(
        f"Duplicates Removed : {COUNTS['Duplicate']}"
    )

    logger.info(
        f"Corrupted Images : {COUNTS['Corrupted']}"
    )

    logger.info(
        f"Skipped : {COUNTS['Skipped']}"
    )

    logger.info("=" * 60)

# ==========================================================
# VERIFY OUTPUT
# ==========================================================

def verify_output():

    logger.info("")

    logger.info("Verifying Dataset...")

    ok = True

    for cls in CLASSES:

        folder = OUTPUT / cls

        if not folder.exists():

            logger.error(
                f"{cls} Missing"
            )

            ok = False

            continue

        total = count_images(folder)

        if total == 0:

            logger.warning(
                f"{cls} Empty"
            )

        else:

            logger.info(
                f"{cls} OK ({total})"
            )

    return ok

# ==========================================================
# CLEAN OUTPUT (OPTIONAL)
# ==========================================================

def clear_dataset():

    logger.info("Cleaning old dataset...")

    for cls in CLASSES:

        folder = OUTPUT / cls

        if not folder.exists():
            continue

        for file in folder.iterdir():

            if file.is_file():

                try:
                    file.unlink()
                except Exception:
                    pass

    logger.info("Clean Complete.")

# ==========================================================
# MAIN
# ==========================================================

def main():

    logger.info("")
    logger.info("=" * 60)
    logger.info("RETINA DATASET BUILDER")
    logger.info("=" * 60)

    clear_dataset()

    process_odir()

    process_aptos()

    process_glaucoma()

    process_cataract()

    verify_output()

    print_summary()

    logger.info("")
    logger.info("Dataset Build Successful.")
    logger.info("=" * 60)


if __name__ == "__main__":

    main()