import os
import json
import random
import shutil
from collections import defaultdict
from tqdm import tqdm

# ---------------- CONFIG ----------------
# k-fold CV
k_folds = 1
seed = 42

# held-out test split at patient level
test_fraction = 0.2   # % of patients held out as test
min_test_patients = 1  # ensure at least 1 test patient if possible
build = os.path.join(os.getcwd(), 'build')
imagesTr = os.path.join(build, "stage_2_training/imagesTr")
labelsTr = os.path.join(build, "stage_2_training/labelsTr")
imagesTs = os.path.join(build, "stage_2_training/testing/imagesTs")
labelsTs = os.path.join(build, 'stage_2_training/testing/labelsTs')

# folders for moved-out cases
testing_root = os.path.join(build, "stage_2_training/testing")
faulty_tr_root = os.path.join(build, "stage_2_trainingg/tr_excluded_labels")
no_label_tr_root = os.path.join(build,"stage_2_training/tr_files_with_no_labels")
no_tr_labels_root = os.path.join(build,"stage_2_training/lbl_files_with_no_tr")
faulty_labels_txt = os.path.join(build,"stage_2_training/faulty_labels.txt")

# Where to write split JSONs
out_dir = os.path.join(build,"stage_2_training/splits")
os.makedirs(out_dir, exist_ok=True)


# ---------------- UTILS ----------------
def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def list_nii_files(folder):
    if not os.path.exists(folder):
        return []
    return [
        f for f in os.listdir(folder)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ]

def get_case_id(path_or_name: str) -> str:
    """
    Case ID = full filename stem (without .nii/.nii.gz).
    e.g. 3128_051_01.nii.gz -> 3128_051_01
    """
    base = os.path.basename(path_or_name.strip())
    if base.endswith(".nii.gz"):
        return base[:-7]
    if base.endswith(".nii"):
        return base[:-4]
    return os.path.splitext(base)[0]

def get_patient_id(path_or_name: str) -> str:
    """
    Patient ID = first two underscore-separated chunks of the case id.
    Examples:
      3128_051_01 -> 3128_051
      4128_043    -> 4128_043
      2128_004_1  -> 2128_004
    """
    cid = get_case_id(path_or_name)
    parts = cid.split("_")
    if len(parts) < 2:
        return cid
    return "_".join(parts[:2])


def read_faulty_case_ids(path):
    ids = set()
    if not os.path.exists(path):
        print(f"[WARN] {path} not found; skipping faulty-image move.")
        return ids

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.add(get_case_id(line))
    return ids

def find_label_filename(cid, labels_dir):
    for ext in (".nii.gz", ".nii"):
        fname = cid + ext
        if os.path.exists(os.path.join(labels_dir, fname)):
            return fname
    return None

def print_counts():
    contrast = 4
    img_count = len(list_nii_files(imagesTr))/contrast 
    lbl_count = len(list_nii_files(labelsTr))

    print(f"imagesTr: {img_count} | labelsTr: {lbl_count}")


# ---------------- RESET (optional but recommended) ----------------
def reset_split():
    """
    Move everything back to training, including any previously created val/test folders
    and moved-out images.
    """

    # moved-out images
    for root in [faulty_tr_root, no_label_tr_root]:
        moved_imgs = os.path.join(root, "imagesTr")
        if os.path.exists(moved_imgs):
            for f in list_nii_files(moved_imgs):
                shutil.move(os.path.join(moved_imgs, f), os.path.join(imagesTr, f))
    
    for root in [no_tr_labels_root]:
        moved_imgs = os.path.join(root, "labelsTr")
        if os.path.exists(moved_imgs):
            for f in list_nii_files(moved_imgs):
                shutil.move(os.path.join(moved_imgs, f), os.path.join(labelsTr, f))

    moved_imgs = os.path.join(testing_root, "imagesTs")   
    if os.path.exists(moved_imgs):
        for f in list_nii_files(moved_imgs):
            shutil.move(os.path.join(moved_imgs, f), os.path.join(imagesTr, f))
           
    moved_imgs = os.path.join(testing_root, "labelsTs")   
    if os.path.exists(moved_imgs):
        for f in list_nii_files(moved_imgs):
            shutil.move(os.path.join(moved_imgs, f), os.path.join(labelsTr, f))
    


    print("Split reset: all files moved back to training")


# ---------------- CLEANUP ----------------
def move_faulty_lbl_tr_images(images_tr_dir, faulty_labels_path, dest_root):
    faulty_ids = read_faulty_case_ids(faulty_labels_path)
    if not faulty_ids:
        return

    dest_images = os.path.join(dest_root, "imagesTr")
    ensure_dirs(dest_images)

    img_files = list_nii_files(images_tr_dir)
    img_map = {get_case_id(f): f for f in img_files}

    moved = 0
    missing = 0

    for cid in sorted(faulty_ids):
        img_name = img_map.get(cid)
        if not img_name:
            missing += 1
            continue
        shutil.move(
            os.path.join(images_tr_dir, img_name),
            os.path.join(dest_images, img_name),
        )
        moved += 1

    final_count = len(list_nii_files(dest_images))
    print(f"Faulty ASL-GT Labels: {final_count} images (moved {moved} this run)")
    if missing:
        print(f"[WARN] {missing} faulty case_ids were not found in {images_tr_dir}; skipped.")

def check_for_label(selected_img, label_ids):
    
    parts = selected_img.split('_')
    
    if len(parts) == 3:
        case_id = f'{parts[0]}_{parts[1]}'
    else:
        case_id = f'{parts[0]}_{parts[1]}_{parts[2]}'

    if case_id in label_ids:
        return selected_img
    else:
        return []


def check_channels(tr_ids):

    
    filtered_ids = {}

    for id in sorted(tr_ids):
        

        parts = id.split('_')
        
        if len(parts) == 3:
            site = parts[0]
            case = parts[1]
            ch = parts[-1]
            key = (site, case)
        else:
            site = parts[0]
            case = parts[1]
            visit = parts[2]
            ch = parts[-1]
            key = (site, case, visit)

            
        filtered_ids[key] = id
    
    incomplete_img = []
    complete_img = []

    filtered_images = list(filtered_ids.values())

    for img in sorted(filtered_images):
        parts = img.split('_')
        site = parts[0]
        case = parts[1]
        ch = parts[-1]
        if len(parts) != 3:
            visit = parts[2]
            img_id = f'{site}_{case}_{visit}'
        else:
            img_id = f'{site}_{case}'
            
        if ch == '0003':

            complete_img.append(img_id)
        else:
            incomplete_img.append(img_id)
    
    return complete_img, incomplete_img
    
def move_incomplete_ch_images(incomplete_ch, images_tr_dir):
    dest_root = os.path.join(os.getcwd(), 'training', 'incomplete_tr')
    dest_images = os.path.join(dest_root, "imagesTr")
    ensure_dirs(dest_images) 

    for img in incomplete_ch:    
        moved = 0 
        shutil.move(
            os.path.join(images_tr_dir, img),
            os.path.join(dest_images, img))

        moved += 1     
    final_count = len(list_nii_files(dest_images))
    print(f"Incomplete images: {final_count} (moved {moved} this run)")


def check_for_image(selected_lbl, images_tr_dir, tr_ids):  

    #check all channels exist for the images
    complete_ch, incomplete_ch = check_channels(tr_ids)

    if incomplete_ch:
        move_incomplete_ch_images(incomplete_ch, images_tr_dir)
    
    if selected_lbl in complete_ch:
        return selected_lbl
    else:
        return []



def move_tr_images_without_label(images_tr_dir, labels_tr_dir, dest_root):
    dest_images = os.path.join(dest_root, "imagesTr")
    ensure_dirs(dest_images)

    lbl_ids = {get_case_id(f) for f in list_nii_files(labels_tr_dir)}

    image_database = os.listdir(images_tr_dir)
    moved = 0
    for img in image_database:

        try:
            lbl_exists = check_for_label(img, lbl_ids)
            
            if lbl_exists:
                continue

            shutil.move(
                os.path.join(images_tr_dir, img),
                os.path.join(dest_images, img))

            moved += 1
        except Exception as e:
            tqdm.write(e)

    final_count = len(list_nii_files(dest_images))
    print(f"No-label images: {final_count} (moved {moved} this run)")

def move_labels_without_tr(images_tr_dir, labels_tr_dir, dest_root):
    dest_labels = os.path.join(dest_root, "labelsTr")
    ensure_dirs(dest_labels)

    tr_ids = {get_case_id(f) for f in list_nii_files(images_tr_dir)}
    
    labels_database = sorted(os.listdir(labels_tr_dir))

    moved = 0
    for lbl in labels_database:
        lbl_id = get_case_id(lbl)
        
        try:
            img_exist = check_for_image(lbl_id, images_tr_dir, tr_ids)
            
            if img_exist:
                continue

            shutil.move(
                os.path.join(labels_tr_dir, lbl),
                os.path.join(dest_labels, lbl),
            )
            moved += 1
        except Exception as e:
            tqdm.write(f'{e}')

    final_count = len(list_nii_files(dest_labels))
    print(f"No-image labels: {final_count} (moved {moved} this run)")


# ---------------- 5-FOLD CV + HELD-OUT TEST JSON GENERATION ----------------
def make_patient_test_split(patients, fraction, rng, min_test=1):
    """
    Returns (test_patients, remaining_patients) using patient-level split.
    """
    patients = list(patients)
    if not patients:
        return [], []

    rng.shuffle(patients)

    # compute test size
    n = len(patients)
    n_test = int(round(n * fraction))
    if n >= 2:
        n_test = max(min_test, n_test)
        n_test = min(n_test, n - 1)  # keep at least 1 patient for CV/train
    else:
        n_test = 0  # can't hold out test if only 1 patient

    test_patients = patients[:n_test]
    remaining = patients[n_test:]
    return test_patients, remaining

def get_id_without_ch(f):
    parts = f.split('_')
    site = parts[0]
    case = parts[1]
    ch = parts[-1]
    if len(parts) != 3:
        visit = parts[2]
        img_id = f'{site}_{case}_{visit}'
    else:
        img_id = f'{site}_{case}'
    return img_id

def move_test_data(test_files):
    images_dir = os.path.join('training', "imagesTr")
    labels_dir = os.path.join('training', "labelsTr")
    dest_images = os.path.join('testing', "imagesTs")
    dest_labels = os.path.join('testing', "labelsTs")
    ensure_dirs(dest_images, dest_labels)

    i_moved = 0
    l_moved = 0
    
    patient_to_files = defaultdict(list)
    for f in test_files:
        pid = get_id_without_ch(f)
        patient_to_files[pid].append(f)
    
    

    lbl_ids = list(patient_to_files)


    
    for tr in test_files:
        try:
            
            shutil.move(
                os.path.join(images_dir, tr),
                os.path.join(dest_images, tr),
            )

            i_moved += 1
        except Exception as e:
            tqdm.write(f'{e}')
    
    for lbl in lbl_ids:
            l = lbl + '.nii.gz'
            shutil.move(
            os.path.join(labels_dir,  l),
            os.path.join(dest_labels, l),
            )   
            l_moved += 1
    contrast = 4 
    final_i_count = len(list_nii_files(dest_images))//contrast
    final_l_count = len(list_nii_files(dest_labels))
    print(f"Total Testing images: {final_i_count} (moved {i_moved} this run)")
    print(f"Total Testing labels: {final_l_count} (moved {i_moved} this run)")

def create_json():
    """
    Generates dataset_fold{0..k-1}.json with:
      - a held-out TEST patient split (constant across folds)
      - k-fold patient-level CV on remaining patients
    Does NOT move files. All images/labels are assumed in imagesTr/labelsTr.
    """
    files = sorted(list_nii_files(imagesTr))
    if not files:
        raise RuntimeError(f"No images found in {imagesTr}")

    # group by patient ID
    patient_to_files = defaultdict(list)
    for f in files:
        pid = get_id_without_ch(f)
        patient_to_files[pid].append(f)

    all_patients = sorted(patient_to_files.keys())
    if len(all_patients) < 2:
        raise ValueError(f"Need at least 2 patients for test+CV; found {len(all_patients)}.")

    # reproducible shuffle/split
    rng = random.Random(seed)
    test_patients, cv_patients = make_patient_test_split(
        all_patients, test_fraction, rng, min_test=min_test_patients
    )

    if len(cv_patients) < k_folds:
        raise ValueError(
            f"After holding out test patients ({len(test_patients)}), "
            f"only {len(cv_patients)} patients remain; cannot do {k_folds}-fold CV."
        )

    # distribute remaining patients into folds as evenly as possible
    rng_cv = random.Random(seed + 999)
    rng_cv.shuffle(cv_patients)
    fold_patients = [cv_patients[i::k_folds] for i in range(k_folds)]

    # write manifest for debugging/repro
    manifest = {
        "k_folds": k_folds,
        "seed": seed,
        "test_fraction": test_fraction,
        "num_patients_total": len(all_patients),
        "num_patients_test": len(test_patients),
        "num_patients_cv_pool": len(cv_patients),
        "test_patients": test_patients,
        "folds": {str(i): fold_patients[i] for i in range(k_folds)},
        "patient_id_format": "id_00_xxxx|id_01_xxxx: precontrast baseline|followup, id_xxxx|id_1_xxxx: postcontrast baseline|followup",
    }
    with open(os.path.join(out_dir, "cv_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


    # build constant test file list (shared for all folds)
    test_files = [f for p in test_patients for f in patient_to_files[p]]
    rng_test = random.Random(seed + 2024)
    rng_test.shuffle(test_files)
    move_test_data(test_files)
    if k_folds == 1:
    # do a simple patient-level train/val split instead of CV
        rng_split = random.Random(seed + 123)
        cv_patients_shuf = cv_patients[:]
        rng_split.shuffle(cv_patients_shuf)

        val_frac = 0.1  # <- choose your ratio
        n_val = max(1, int(round(len(cv_patients_shuf) * val_frac)))
        n_val = min(n_val, len(cv_patients_shuf) - 1)  # keep at least 1 train patient

        fold_patients = [cv_patients_shuf[:n_val]]
        # and redefine cv_patients pool for training as the remainder
        cv_patients = cv_patients_shuf[n_val:] + cv_patients_shuf[:n_val]  # keep full list if you want later

    # build each fold json
    for fold in range(k_folds):
        val_patients = set(fold_patients[fold])
        train_patients = set(cv_patients) - val_patients
        assert train_patients.isdisjoint(val_patients)

        train_files = [f for p in train_patients for f in patient_to_files[p]]
        val_files = [f for p in val_patients for f in patient_to_files[p]]

        # optional: shuffle order inside split (reproducible per fold)
        rng_fold = random.Random(seed + fold)
        rng_fold.shuffle(train_files)
        rng_fold.shuffle(val_files)

        train_images = [get_id_without_ch(f) for f in train_files]
        test_images = [get_id_without_ch(f) for f in test_files]
        val_images = [get_id_without_ch(f) for f in val_files]

        
        train_total = len(train_images) + len(val_images)

        dataset = {
            "name": "iBEAt 3D Kidney Segmentation Dixon 4-Channel",
            "licence": "apache",
            "reference": "Sheffield University",
            "release": "x.x xx/xx/xxxx",
            "tensorImageSize": "3D",
            "modality": {"0": "MRI"},
            "numTraining": train_total,
            "numTest": len(test_images),

            "channel_names": {
                "0": "outphase",
                "1": "inphase",
                "2": "fat",
                "3": "water",
            },
            "labels": {
                "background": 0,
                "LK": 1,
                "RK": 2
            },
            "file_ending": ".nii.gz"
        }
        

        out_path = os.path.join(out_dir, f"dataset.json") 
        with open(out_path, "w") as f:
            json.dump(dataset, f, indent=4)

        print(
            f"total train|test: {train_total}|{len(test_files)}"
            f"total patients images train|test: {len(train_images)}|{len(test_images)}"
        )

    print(f"Done. Use dataset.json for CV runs (with a fixed held-out test split).")


def reset_all():
    ensure_dirs(imagesTr, labelsTr)

    # recommended: make sure everything is back in imagesTr/labelsTr before building folds
    reset_split()
    print_counts()


# ---------------- BUILD FOLDS ----------------
def create_fold_0_database():
    ensure_dirs(imagesTr, labelsTr, imagesTs, labelsTs)

    # recommended: make sure everything is back in imagesTr/labelsTr before building folds
    reset_split()
    print_counts()

    # cleanup before CV
    move_faulty_lbl_tr_images(imagesTr, faulty_labels_txt, faulty_tr_root)
    move_tr_images_without_label(imagesTr, labelsTr, no_label_tr_root)
    move_labels_without_tr(imagesTr, labelsTr, no_tr_labels_root)
    print_counts()

    # generate 5-fold CV + held-out TEST dataset JSONs (patient-level, suffix-agnostic)
    create_json()

    files = list_nii_files(imagesTr)

    patient_to_images = defaultdict(list)
    for f in files:
        pid = get_patient_id(f)   # first two underscore chunks
        patient_to_images[pid].append(f)

    # Summary stats
    counts = [len(v) for v in patient_to_images.values()]
    print("\nSummary:")
    print(f"  Num patients: {len(counts)}")
    print(f"  Min images/patient: {min(counts)}")
    print(f"  Max images/patient: {max(counts)}")
    print(f"  Mean images/patient: {sum(counts)//len(counts)}")

def run(build):       

    create_fold_0_database(build)

    #OPTIONAL: reset images+labels to training folder
    #reset_all()
if __name__ == "__main__":
    build = ""
    run(build)
