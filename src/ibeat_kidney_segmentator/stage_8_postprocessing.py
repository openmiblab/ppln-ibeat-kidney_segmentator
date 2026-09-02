import os
import glob
import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm


def isolate_largest_kidney_components(
        input_dir,
        output_dir
):
    """
    Isolate the largest connected component for each kidney label.

    Label convention:
        1 = Left Kidney (LK)
        2 = Right Kidney (RK)

    All other labels are set to 0.
    """

    os.makedirs(output_dir, exist_ok=True)

    nifti_files = glob.glob(os.path.join(input_dir, "*.nii.gz"))

    for nifti_file in tqdm(nifti_files, desc='Post-processing', unit='case'):

        filename = os.path.basename(nifti_file)
        cid = filename.replace('.nii.gz', '')
        filepath = os.path.join(output_dir, filename)
        if cid in VALIDATION:
            tqdm.write(f'Skipping {cid}. gt in val')
            continue

        if os.path.exists(filepath):
            tqdm.write(f'case {cid} in file')
            continue

        


        # --------------------------------------------------
        # Load segmentation
        # --------------------------------------------------

        nii = nib.load(nifti_file)
        data = nii.get_fdata()

        # Preserve original datatype where possible
        output = np.zeros_like(data, dtype=np.uint8)

        # --------------------------------------------------
        # Process each kidney label independently
        # --------------------------------------------------

        for label in [1, 2]:

            # Binary mask for current kidney
            binary_mask = data == label

            if not np.any(binary_mask):
                continue

            # Connected-component labelling
            labelled_mask, num_components = ndimage.label(
                binary_mask,
                structure=np.ones((3, 3, 3), dtype=np.uint8)
            )

            # Find component sizes
            component_sizes = np.bincount(
                labelled_mask.ravel()
            )

            # Ignore background
            component_sizes[0] = 0

            # Largest component
            largest_component = np.argmax(component_sizes)

            # Keep the largest component with its original label
            output[labelled_mask == largest_component] = label

        # --------------------------------------------------
        # Save output
        # --------------------------------------------------

        output_nii = nib.Nifti1Image(
            output,
            affine=nii.affine,
            header=nii.header
        )

        output_path = os.path.join(
            output_dir,
            filename
        )

        nib.save(output_nii, output_path)

        print(f"Saved: {output_path}")

VALIDATION = [
    "1128_005",
    "1128_018_00",
    "1128_019",
    "1128_044",
    "1128_047",
    "1128_057",
    "1128_059",
    "1128_066_00",
    "1128_080",
    "1128_082_00",
    "1128_088",
    "1128_096",
    "1128_C01_V2",
    "2128_005_1",
    "2128_006_1",
    "2128_008",
    "2128_012",
    "2128_021_1",
    "2128_023",
    "2128_025",
    "2128_029",
    "2128_030_1",
    "2128_031",
    "2128_031_1",
    "2128_036",
    "2128_045",
    "3128_007_1",
    "3128_019",
    "3128_021",
    "3128_023",
    "3128_031_1",
    "3128_035_1",
    "3128_041_1",
    "3128_049",
    "3128_053",
    "3128_064_1",
    "3128_067",
    "3128_067_1",
    "3128_073",
    "3128_078",
    "3128_081",
    "3128_085",
    "3128_090",
    "3128_090_1",
    "3128_093",
    "3128_102",
    "3128_111",
    "3128_113",
    "3128_133",
    "3128_C01_V1",
    "3128_C02_V1",
    "4128_008",
    "4128_014",
    "4128_035",
    "4128_038",
    "4128_041",
    "4128_048_00",
    "4128_C13_V1",
    "4128_C16_V1",
    "4128_C22_V1",
    "5128_014",
    "5128_025",
    "5128_028",
    "5128_034",
    "5128_042",
    "5128_043_1",
    "5128_051",
    "5128_055",
    "5128_056",
    "5128_057",
    "5128_069",
    "5128_071",
    "5128_072",
    "5128_073",
    "5128_075",
    "5128_085_1",
    "5128_086",
    "5128_088",
    "5128_095",
    "5128_098",
    "5128_C02_V3",
    "5128_C02_V4",
    "5128_C03_V2",
    "5128_C10_V2",
    "5128_C11_V1",
    "6128_005_00",
    "7128_003_00",
    "7128_024",
    "7128_035",
    "7128_059",
    "7128_066",
    "7128_067",
    "7128_069",
    "7128_073",
    "7128_080",
    "7128_083",
    "7128_090_00",
    "7128_092",
    "7128_096",
    "7128_100",
    "7128_103",
    "7128_111",
    "7128_131_00",
    "7128_132",
    "7128_134",
    "7128_136_00",
    "7128_137",
    "7128_140",
    "7128_144",
    "7128_146_00",
    "7128_147",
    "7128_158_00",
    "7128_163",
    "7128_165_00"]
# ==========================================================
# Postprocess NifTi inference data 
# ==========================================================
if __name__ == '__main__':

    input_dir = r""
    output_dir = r""

    isolate_largest_kidney_components(
        input_dir=input_dir,
        output_dir=output_dir
    )