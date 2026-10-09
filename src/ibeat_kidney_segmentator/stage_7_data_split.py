
import os
import json
import random
import shutil
from collections import defaultdict


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
k_folds = 1
seed = 42

# Held-out test split at patient level
test_fraction = 0.2
validation_fraction = 0.1
min_test_patients = 1

# Keep the original configured roots
build = os.path.join(
    os.getcwd(), "iBEAt_Build", "kidney_segmentation"
)

imagesTr = os.path.join(
    build, "stage_6_data_prep/nnunet_raw/Dataset001_Kidneys/imagesTr"
)
labelsTr = os.path.join(
    build, "stage_6_data_prep/nnunet_raw/Dataset001_Kidneys/labelsTr"
)
imagesTs = os.path.join(
    build, "stage_6_data_prep/nnunet_raw/Dataset001_Kidneys/testing/imagesTs"
)
labelsTs = os.path.join(
    build, "stage_6_data_prep/nnunet_raw/Dataset001_Kidneys/testing/labelsTs"
)

# Folders for excluded cases
testing_root = os.path.join(
    build, "stage_6_data_prep/nnunet_raw/Dataset001_Kidneys/testing"
)
faulty_tr_root = os.path.join(
    build, "stage_6_data_prep/nnunet_raw/tr_excluded_labels"
)
no_label_tr_root = os.path.join(
    build, "stage_6_data_prep/nnunet_raw/tr_img_with_no_labels"
)
no_tr_labels_root = os.path.join(
    build, "stage_6_data_prep/nnunet_raw/lbls_with_no_img"
)
faulty_labels_txt = os.path.join(
    build, "stage_6_data_prep/nnunet_raw/faulty_labels.txt"
)

# Preserve the original incomplete-channel folder
incomplete_root = os.path.join(
    os.getcwd(), "training", "incomplete_tr"
)

# JSON output folder
out_dir = os.path.join(
    build, "stage_6_data_prep/nnunet_raw/Dataset001_Kidneys"
)
os.makedirs(out_dir, exist_ok=True)

# Expected Dixon channels
CHANNELS = {"0000", "0001", "0002", "0003"}

CHANNEL_NAMES = {
    "0": "outphase",
    "1": "inphase",
    "2": "fat",
    "3": "water",
}


# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------
def ensure_dirs(*dirs):
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def list_nii_files(folder):
    if not os.path.isdir(folder):
        return []

    return sorted(
        f for f in os.listdir(folder)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    )


def get_case_id(path_or_name):
    """Remove the .nii or .nii.gz extension."""
    name = os.path.basename(str(path_or_name).strip())

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return os.path.splitext(name)[0]


def get_case_and_channel(path_or_name):
    """
    Example:
        2128_C01_V1_0003.nii.gz
        -> ("2128_C01_V1", "0003")
    """
    case_id = get_case_id(path_or_name)
    parts = case_id.rsplit("_", 1)

    if len(parts) == 2 and parts[1] in CHANNELS:
        return parts[0], parts[1]

    return case_id, None


def get_id_without_ch(path_or_name):
    return get_case_and_channel(path_or_name)[0]


def get_patient_id(path_or_name):
    """
    Preserve the original patient ID convention:
    first two underscore-separated fields.
    """
    case_id = get_id_without_ch(path_or_name)
    parts = case_id.split("_")

    if len(parts) < 2:
        return case_id

    return "_".join(parts[:2])


def image_map(folder):
    """
    Return:
        {case_id: {channel: filename}}
    """
    result = defaultdict(dict)

    for filename in list_nii_files(folder):
        case_id, channel = get_case_and_channel(filename)

        if channel is not None:
            result[case_id][channel] = filename

    return dict(result)


def label_map(folder):
    """
    Return:
        {case_id: label_filename}

    Supports both .nii.gz and .nii labels.
    """
    return {
        get_case_id(filename): filename
        for filename in list_nii_files(folder)
    }


def find_label_filename(case_id, folder):
    return label_map(folder).get(case_id)


def move_file(src, dst):
    """Move an existing file without silently overwriting another file."""
    if not os.path.exists(src):
        return False

    ensure_dirs(os.path.dirname(dst))

    if os.path.exists(dst):
        raise FileExistsError(
            f"Destination already exists:\n{dst}\n"
            f"Source:\n{src}"
        )

    shutil.move(src, dst)
    return True


def read_faulty_case_ids(path):
    ids = set()

    if not os.path.exists(path):
        print(
            f"[WARN] {path} not found; "
            "skipping faulty-image exclusion."
        )
        return ids

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            ids.add(get_case_id(line))

    return ids


def print_counts():
    img_files = list_nii_files(imagesTr)
    case_count = len(image_map(imagesTr))
    label_count = len(list_nii_files(labelsTr))

    print(
        f"imagesTr: {len(img_files)} channel files / "
        f"{case_count} cases | labelsTr: {label_count}"
    )

    print(
        f"imagesTs: {len(list_nii_files(imagesTs))} channel files | "
        f"labelsTs: {len(list_nii_files(labelsTs))}"
    )


# ---------------------------------------------------------------------
# RESET PREVIOUS SPLITS
# ---------------------------------------------------------------------
def restore_folder(source_dir, destination_dir):
    """Restore files from a previous split if they are not duplicates."""
    for filename in list_nii_files(source_dir):
        src = os.path.join(source_dir, filename)
        dst = os.path.join(destination_dir, filename)

        if os.path.exists(dst):
            print(f"[WARN] Duplicate not restored: {dst}")
            continue

        shutil.move(src, dst)


def reset_split():
    """
    Restore previously moved test and excluded files to the
    configured imagesTr and labelsTr directories.
    """
    ensure_dirs(imagesTr, labelsTr, imagesTs, labelsTs)

    # Restore held-out test data.
    restore_folder(imagesTs, imagesTr)
    restore_folder(labelsTs, labelsTr)

    # Restore excluded images.
    for root in (faulty_tr_root, no_label_tr_root):
        restore_folder(
            os.path.join(root, "imagesTr"),
            imagesTr
        )

    # Restore labels without corresponding images.
    restore_folder(
        os.path.join(no_tr_labels_root, "labelsTr"),
        labelsTr
    )

    # Restore incomplete-channel images.
    restore_folder(
        os.path.join(incomplete_root, "imagesTr"),
        imagesTr
    )

    print("Split reset complete.")
    print_counts()


# ---------------------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------------------
def move_faulty_case_images():
    """
    Exclude image channels associated with case IDs listed in
    faulty_labels.txt.
    """
    faulty_ids = read_faulty_case_ids(faulty_labels_txt)

    if not faulty_ids:
        return

    dest_images = os.path.join(faulty_tr_root, "imagesTr")
    ensure_dirs(dest_images)

    moved = 0

    for filename in list_nii_files(imagesTr):
        case_id, _ = get_case_and_channel(filename)
        full_id = get_case_id(filename)

        if case_id in faulty_ids or full_id in faulty_ids:
            src = os.path.join(imagesTr, filename)
            dst = os.path.join(dest_images, filename)

            if move_file(src, dst):
                moved += 1

    print(f"Faulty-label images: moved {moved} channel files.")


def move_incomplete_channel_cases():
    """
    Move all available channels for cases that do not have all four
    expected Dixon channels.
    """
    cases = image_map(imagesTr)
    dest_images = os.path.join(incomplete_root, "imagesTr")
    ensure_dirs(dest_images)

    incomplete_cases = {
        case_id: channels
        for case_id, channels in cases.items()
        if set(channels) != CHANNELS
    }

    moved = 0

    for case_id, channels in sorted(incomplete_cases.items()):
        missing = sorted(CHANNELS - set(channels))

        print(
            f"[WARN] Incomplete channels for {case_id}; "
            f"missing {missing}"
        )

        for filename in channels.values():
            src = os.path.join(imagesTr, filename)
            dst = os.path.join(dest_images, filename)

            if move_file(src, dst):
                moved += 1

    print(f"Incomplete-channel images: moved {moved} files.")


def move_images_without_label():
    """Move all four channels when a case has no matching label."""
    dest_images = os.path.join(no_label_tr_root, "imagesTr")
    ensure_dirs(dest_images)

    labels = label_map(labelsTr)
    cases = image_map(imagesTr)

    moved = 0

    for case_id, channels in sorted(cases.items()):
        if case_id in labels:
            continue

        for filename in channels.values():
            src = os.path.join(imagesTr, filename)
            dst = os.path.join(dest_images, filename)

            if move_file(src, dst):
                moved += 1

    print(f"Images without labels: moved {moved} channel files.")


def move_labels_without_image():
    """Move labels that do not have a matching four-channel case."""
    dest_labels = os.path.join(no_tr_labels_root, "labelsTr")
    ensure_dirs(dest_labels)

    cases = set(image_map(imagesTr))
    labels = label_map(labelsTr)

    moved = 0

    for case_id, filename in labels.items():
        if case_id in cases:
            continue

        src = os.path.join(labelsTr, filename)
        dst = os.path.join(dest_labels, filename)

        if move_file(src, dst):
            moved += 1

    print(f"Labels without images: moved {moved} labels.")


def clean_dataset():
    move_faulty_case_images()
    move_incomplete_channel_cases()
    move_images_without_label()
    move_labels_without_image()


# ---------------------------------------------------------------------
# PATIENT-LEVEL TEST SPLIT
# ---------------------------------------------------------------------
def make_patient_test_split(
    patients,
    fraction,
    rng,
    min_test=1
):
    patients = list(patients)

    if not patients:
        return [], []

    rng.shuffle(patients)

    n = len(patients)
    n_test = int(round(n * fraction))

    if n >= 2:
        n_test = max(min_test, n_test)
        n_test = min(n_test, n - 1)
    else:
        n_test = 0

    return patients[:n_test], patients[n_test:]


# ---------------------------------------------------------------------
# MOVE TEST DATA
# ---------------------------------------------------------------------
def move_test_data(test_files):
    """
    Move test images from imagesTr to the configured testing/imagesTs
    folder and corresponding labels to testing/labelsTs.
    """
    ensure_dirs(imagesTr, labelsTr, imagesTs, labelsTs)

    test_case_ids = sorted({
        get_id_without_ch(filename)
        for filename in test_files
    })

    moved_images = 0
    moved_labels = 0
    missing_images = []
    missing_labels = []

    # Move image channels.
    for filename in test_files:
        src = os.path.join(imagesTr, filename)
        dst = os.path.join(imagesTs, filename)

        if os.path.exists(src):
            if os.path.exists(dst):
                raise FileExistsError(
                    f"Test image already exists: {dst}. "
                    "Reset the split before rerunning."
                )

            shutil.move(src, dst)
            moved_images += 1

        elif not os.path.exists(dst):
            missing_images.append(filename)

    # Move one matching label per case.
    for case_id in test_case_ids:
        label_filename = find_label_filename(case_id, labelsTr)

        if label_filename is None:
            # Check whether it was already moved in a previous run.
            if find_label_filename(case_id, labelsTs) is None:
                missing_labels.append(case_id)
            continue

        src = os.path.join(labelsTr, label_filename)
        dst = os.path.join(labelsTs, label_filename)

        if os.path.exists(dst):
            raise FileExistsError(
                f"Test label already exists: {dst}. "
                "Reset the split before rerunning."
            )

        shutil.move(src, dst)
        moved_labels += 1

    print(f"Total Testing image files moved: {moved_images}")
    print(f"Total Testing labels moved: {moved_labels}")

    if missing_images:
        print("[WARN] Missing test images:")
        for filename in missing_images:
            print(f"  {filename}")

    if missing_labels:
        print("[WARN] Missing test labels:")
        for case_id in missing_labels:
            print(f"  {case_id}")


# ---------------------------------------------------------------------
# GENERATE DATASET JSON AND SPLIT MANIFEST
# ---------------------------------------------------------------------
def create_json():
    files = list_nii_files(imagesTr)
    cases = image_map(imagesTr)

    if not files or not cases:
        raise RuntimeError(f"No images found in {imagesTr}")

    labels = label_map(labelsTr)

    # Only complete, labelled cases enter the split.
    eligible_cases = sorted(
        case_id
        for case_id, channels in cases.items()
        if set(channels) == CHANNELS and case_id in labels
    )

    if len(eligible_cases) < 2:
        raise ValueError(
            "Need at least two complete labelled cases; "
            f"found {len(eligible_cases)}."
        )

    # Group cases by patient ID.
    patient_to_cases = defaultdict(list)

    for case_id in eligible_cases:
        patient_to_cases[get_patient_id(case_id)].append(case_id)

    all_patients = sorted(patient_to_cases)

    if len(all_patients) < 2:
        raise ValueError(
            "Need at least two patients for a test/train split; "
            f"found {len(all_patients)}."
        )

    rng = random.Random(seed)

    test_patients, trainval_patients = make_patient_test_split(
        all_patients,
        test_fraction,
        rng,
        min_test=min_test_patients
    )

    if not trainval_patients:
        raise ValueError("No training patients remain after test split.")

    # Patient-level validation split.
    rng_val = random.Random(seed + 123)
    shuffled_patients = trainval_patients[:]
    rng_val.shuffle(shuffled_patients)

    if len(shuffled_patients) >= 2:
        n_val = max(
            1,
            int(round(len(shuffled_patients) * validation_fraction))
        )
        n_val = min(n_val, len(shuffled_patients) - 1)

        val_patients = set(shuffled_patients[:n_val])
        train_patients = set(shuffled_patients[n_val:])
    else:
        val_patients = set()
        train_patients = set(shuffled_patients)

    test_case_ids = sorted(
        case_id
        for patient in test_patients
        for case_id in patient_to_cases[patient]
    )

    train_case_ids = sorted(
        case_id
        for patient in train_patients
        for case_id in patient_to_cases[patient]
    )

    val_case_ids = sorted(
        case_id
        for patient in val_patients
        for case_id in patient_to_cases[patient]
    )

    # Save the split manifest.
    manifest = {
        "k_folds": k_folds,
        "seed": seed,
        "test_fraction": test_fraction,
        "validation_fraction": validation_fraction,
        "num_patients_total": len(all_patients),
        "num_patients_test": len(test_patients),
        "num_patients_train": len(train_patients),
        "num_patients_validation": len(val_patients),
        "test_patients": sorted(test_patients),
        "train_patients": sorted(train_patients),
        "validation_patients": sorted(val_patients),
        "patient_id_format": "first two underscore-separated fields",
    }

    with open(
        os.path.join(out_dir, "cv_manifest.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    # Build the list of test channel images.
    test_files = [
        cases[case_id][channel]
        for case_id in test_case_ids
        for channel in sorted(CHANNELS)
    ]

    # Move the held-out test images and labels.
    move_test_data(test_files)

    # Preserve the original dataset JSON structure.
    dataset = {
        "name": "iBEAt 3D Kidney Segmentation Dixon 4-Channel",
        "licence": "apache",
        "reference": "Sheffield University",
        "release": "x.x xx/xx/xxxx",
        "tensorImageSize": "3D",
        "modality": {"0": "MRI"},
        "numTraining": len(train_case_ids) + len(val_case_ids),
        "numTest": len(test_case_ids),
        "channel_names": CHANNEL_NAMES,
        "labels": {
            "background": 0,
            "LK": 1,
            "RK": 2
        },
        "file_ending": ".nii.gz"
    }

    with open(
        os.path.join(out_dir, "dataset.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(dataset, f, indent=4)

    print("\nSplit summary:")
    print(f"  Eligible cases: {len(eligible_cases)}")
    print(f"  Patients total: {len(all_patients)}")
    print(f"  Train cases: {len(train_case_ids)}")
    print(f"  Validation cases: {len(val_case_ids)}")
    print(f"  Test cases: {len(test_case_ids)}")
    print(f"  Test image files: {len(test_files)}")
    print(f"  Test labels folder: {labelsTs}")
    print(f"  Dataset JSON: {os.path.join(out_dir, 'dataset.json')}")


# ---------------------------------------------------------------------
# BUILD DATABASE
# ---------------------------------------------------------------------
def create_fold_database():
    ensure_dirs(imagesTr, labelsTr, imagesTs, labelsTs)

    # Restore files from a previous split before rebuilding.
    reset_split()

    print("\nBefore cleanup:")
    print_counts()

    # Check faulty cases and missing/incomplete data.
    clean_dataset()

    print("\nAfter cleanup:")
    print_counts()

    # Generate the patient-level split and move test files.
    create_json()

    # ---------------------------------------------------------------
    # ORIGINAL SUMMARY FORMAT
    # ---------------------------------------------------------------
    files = list_nii_files(imagesTr)

    patient_to_images = defaultdict(list)

    for f in files:
        pid = get_patient_id(f)
        patient_to_images[pid].append(f)

    counts = [len(v) for v in patient_to_images.values()]

    print("\nSummary:")
    print(f"  Num patients: {len(counts)}")

    if counts:
        print(f"  Min images/patient: {min(counts)}")
        print(f"  Max images/patient: {max(counts)}")
        print(f"  Mean images/patient: {sum(counts)//len(counts)}")


def reset_all():
    ensure_dirs(imagesTr, labelsTr)
    reset_split()
    print_counts()


def run():
    create_fold_database()


if __name__ == "__main__":
    run()
