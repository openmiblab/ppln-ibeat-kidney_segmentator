# iBEAtSegmentator model development

## Stages

- stage_1_build_canvas: Takes clean dixon data as input, creates canvas and saves as DICOM
- stage_2_wezel_segmentation: offline (currently blank): takes canvas + clean dixon data and creates kidney segmentations with a dedicated tool in wezel, saved as raw data folder in dicom
- stage_3_clean_data: restructures and organises the wezel segementations from step 2 into a harmonized dicom folder structure.
- stage_4_view_and_edit: etc....