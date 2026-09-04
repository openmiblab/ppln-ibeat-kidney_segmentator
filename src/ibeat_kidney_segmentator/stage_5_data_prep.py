import os
import dbdicom as db
from tqdm import tqdm 
import numpy as np
import nibabel as nib
import utils.data as data

def check_precontrast_ids(input_case_id, study_type):   
        if study_type == 'Baseline':
            
            precontrast_cases =[
                '1128_018',
                '1128_032',
                '1128_045',
                '1128_061',
                '1128_066',
                '1128_067',
                '1128_068',
                '1128_075',
                '1128_081',
                '1128_082',
                '1128_083',
                '2128_043',
                '3128_017',
                '3128_034',
                '3128_063',
                '3128_064',
                '3128_119',
                '4128_013',
                '4128_028',
                '4128_045',
                '4128_048',
                '5128_061',
                '5128_066',
                '5128_101',
                '6128_005',
                '6128_006',
                '6128_009',
                '7128_003',
                '7128_004',
                '7128_025',
                '7128_043',
                '7128_046',
                '7128_025',
                '7128_043',
                '7128_046',
                '7128_051',
                '7128_060',
                '7128_064',
                '7128_078',
                '7128_079',
                '7128_081',
                '7128_090',
                '7128_095',
                '7128_099',
                '7128_107',
                '7128_108',
                '7128_120',
                '7128_126',
                '7128_127',
                '7128_128',
                '7128_130',
                '7128_131',
                '7128_135',
                '7128_136',
                '7128_142',
                '7128_143',
                '7128_146',
                '7128_151',
                '7128_152',
                '7128_153',
                '7128_158',
                '7128_159',
                '7128_165'
            ]
        else:
            precontrast_cases = [
            
            '2128_002',
            '5128_068',
            '6128_008'
            ]

        if input_case_id in precontrast_cases:
            output_case_id = input_case_id
        else:
            output_case_id = []
        
        return output_case_id

def create_database(
    series,
    case_idx=1,
    visit_idx=2,
    series_idx=3,
    include_substring=None,
    exclude_substring=None,
):
    latest = {}

    for i in series:
        series_name = i[series_idx][0]

        # include filter (if set, must match)
        if include_substring and include_substring not in series_name:
            continue

        # exclude filter (if set, must NOT match)
        if exclude_substring and exclude_substring in series_name:
            continue

        case_id = i[case_idx]
        visit   = i[visit_idx][0]   # "baseline", "followup"
        #scan    = series_name       # pre, post-contrast, etc.

        key = (case_id, visit)
        latest[key] = i   # later case overrides earlier case *within same visit*

    return list(latest.values())


def prep_tr_data_nnunet(site=None, study_type=None, cohort='Controls', precontrast=False, visit=1):
    # build table_dir
    if site is not None:
        data_dir = os.path.join(os.getcwd(), 'build', "stage_0_restored_data", 'dixons', cohort, site)
    else:
        data_dir = os.path.join(os.getcwd(), 'build', "stage_0_restored_data", "dixon", cohort)

    dest_dir_lbls = os.path.join(os.getcwd(), 'build', "stage_2_training", "nnunet_raw", "imagesTr")
    os.makedirs(dest_dir_lbls, exist_ok=True)

    if cohort == 'Controls':
        outphase_dir = create_database(db.series(data_dir, contains='out_phase'))
        outphase_dir = [i for i in outphase_dir if i[2][0] == f'Visit{visit}']
        inphase_dir = create_database(db.series(data_dir, contains='in_phase'))
        inphase_dir = [i for i in inphase_dir if i[2][0] == f'Visit{visit}']
        fat_dir = create_database(db.series(data_dir, contains='fat'))
        fat_dir = [i for i in fat_dir if i[2][0] == f'Visit{visit}']
        water_dir = create_database(db.series(data_dir, contains='water'))
        water_dir = [i for i in water_dir if i[2][0] == f'Visit{visit}']  
    else:
        if precontrast == True:
            outphase_dir = create_database(db.series(data_dir, contains='out_phase'), exclude_substring='post_contrast')
            inphase_dir = create_database(db.series(data_dir, contains='in_phase'), exclude_substring='post_contrast')
            fat_dir = create_database(db.series(data_dir, contains='fat'), exclude_substring='post_contrast')
            water_dir = create_database(db.series(data_dir, contains='water'), exclude_substring='post_contrast')
        else:  
            outphase_dir = create_database(db.series(data_dir, contains='out_phase'), include_substring='post_contrast')
            inphase_dir = create_database(db.series(data_dir, contains='in_phase'), include_substring='post_contrast')
            fat_dir = create_database(db.series(data_dir, contains='fat'), include_substring='post_contrast')
            water_dir = create_database(db.series(data_dir, contains='water'), include_substring='post_contrast')

        if study_type is not None:
            outphase_dir = create_database([i for i in outphase_dir if i[2][0] == study_type])
            inphase_dir = create_database([i for i in inphase_dir if i[2][0] == study_type])
            fat_dir = create_database([i for i in fat_dir if i[2][0] == study_type])
            water_dir = create_database([i for i in water_dir if i[2][0] == study_type])


    merged = []
    desc_site = site if site else "Controls"
    # List of selected dixon series
    record = data.dixon_record(parent='ibeat_kidney_shape')

    for outphase in tqdm(sorted(outphase_dir), desc=f'Processing {desc_site} (study/visit: {study_type if study_type is not None else visit}) DICOMs to Nifti', unit='case') :
        
        case_id = outphase[1]
        if case_id not in ('1128_069'):
            continue
            
        if cohort != 'Controls':
            its_precontrast = check_precontrast_ids(case_id, study_type)

            if its_precontrast:
                # this ID belongs to a precontrast image
                if study_type == "Followup":
                    o_output_path = os.path.join(dest_dir_lbls, f"{case_id}_01_0000.nii.gz")
                    i_output_path = os.path.join(dest_dir_lbls, f"{case_id}_01_0001.nii.gz")
                    f_output_path = os.path.join(dest_dir_lbls, f"{case_id}_01_0002.nii.gz")
                    w_output_path = os.path.join(dest_dir_lbls, f"{case_id}_01_0003.nii.gz")
                else:
                    o_output_path = os.path.join(dest_dir_lbls, f"{case_id}_00_0000.nii.gz")
                    i_output_path = os.path.join(dest_dir_lbls, f"{case_id}_00_0001.nii.gz")
                    f_output_path = os.path.join(dest_dir_lbls, f"{case_id}_00_0002.nii.gz")
                    w_output_path = os.path.join(dest_dir_lbls, f"{case_id}_00_0003.nii.gz")

            else:
                # this ID belongs to a postcontrast image
                if study_type == "Followup":
                    o_output_path = os.path.join(dest_dir_lbls, f"{case_id}_1_0000.nii.gz")
                    i_output_path = os.path.join(dest_dir_lbls, f"{case_id}_1_0001.nii.gz")
                    f_output_path = os.path.join(dest_dir_lbls, f"{case_id}_1_0002.nii.gz")
                    w_output_path = os.path.join(dest_dir_lbls, f"{case_id}_1_0003.nii.gz")
                else:
                    o_output_path = os.path.join(dest_dir_lbls, f"{case_id}_0000.nii.gz")
                    i_output_path = os.path.join(dest_dir_lbls, f"{case_id}_0001.nii.gz")
                    f_output_path = os.path.join(dest_dir_lbls, f"{case_id}_0002.nii.gz")
                    w_output_path = os.path.join(dest_dir_lbls, f"{case_id}_0003.nii.gz")
        
        else:
            o_output_path = os.path.join(dest_dir_lbls, f"{case_id}_V{visit}_0000.nii.gz")
            i_output_path = os.path.join(dest_dir_lbls, f"{case_id}_V{visit}_0001.nii.gz")
            f_output_path = os.path.join(dest_dir_lbls, f"{case_id}_V{visit}_0002.nii.gz")
            w_output_path = os.path.join(dest_dir_lbls, f"{case_id}_V{visit}_0003.nii.gz")
        
        if os.path.exists(o_output_path):
            if cohort == 'Controls':
                tqdm.write(f'{case_id}_V{visit} NiFTis exists in directory, skipping')
            else:
                if its_precontrast:
                    tqdm.write(f'{case_id} precontrast {study_type} NiFTis exists in directory, skipping')
                else:
                    tqdm.write(f'{case_id} postcontrast {study_type} NiFTis exists in directory, skipping')
            continue

        
        study = outphase[2][0]
        
        # corresponsing images for the same case
        inphase = next(i for i in inphase_dir if i[1] == case_id)
        fat = next(i for i in fat_dir if i[1] == case_id)
        water = next(i for i in water_dir if i[1] == case_id)

        series_op_desc = outphase[3][0]
        series_in_desc = inphase[3][0]
        series_fat_desc = fat[3][0]
        series_water_desc = water[3][0]

        o_sequence = series_op_desc[:-10]
        i_sequence = series_in_desc[:-9]
        f_sequence = series_fat_desc[:-4]
        w_sequence = series_water_desc[:-6]

        selected_sequence = data.dixon_series_desc(record, case_id, study)
        
        if o_sequence != selected_sequence:
            tqdm.write(f'{case_id} o seq {o_sequence} does not match records seq {selected_sequence}, skipping!')
            continue
        elif i_sequence != selected_sequence:
            tqdm.write(f'{case_id} i seq {i_sequence} does not match records seq {selected_sequence}, skipping!')
            continue
        elif f_sequence != selected_sequence:
            tqdm.write(f'{case_id} f seq {f_sequence} does not match records seq {selected_sequence}, skipping!')
            continue
        elif w_sequence != selected_sequence:
            tqdm.write(f'{case_id} w seq {w_sequence} does not match records seq {selected_sequence}, skipping!')
            continue
        
        


        try:
            outphase = db.volume(outphase)
            inphase = db.volume(inphase)
            fat = db.volume(fat)
            water = db.volume(water)


        except Exception as e:
            if cohort != 'Controls':
                tqdm.write(f'Cannot load {case_id} {study_type}: {e}')
            else:
                tqdm.write(f'Cannot load {case_id} {visit}: {e}')
            continue
    

        outphase_vol = outphase.values
        inphase_vol = inphase.values
        fat_vol = fat.values
        water_vol = water.values
        
        out_affine = outphase.affine
        in_affine = inphase.affine
        fat_affine = fat.affine
        water_affine = water.affine
        
        o_nii_img = nib.Nifti1Image(outphase_vol, out_affine)
        i_nii_img = nib.Nifti1Image(inphase_vol, in_affine)
        f_nii_img = nib.Nifti1Image(fat_vol, fat_affine)
        w_nii_img = nib.Nifti1Image(water_vol, water_affine)


        

        nib.save(o_nii_img, o_output_path)
        nib.save(i_nii_img, i_output_path)
        nib.save(f_nii_img, f_output_path)
        nib.save(w_nii_img, w_output_path)

        print(f'Saved Nifti in folder {o_output_path}')
        print(f'Saved Nifti in folder {i_output_path}')
        print(f'Saved Nifti in folder {f_output_path}')
        print(f'Saved Nifti in folder {w_output_path}')

def prep_labels(site=None, study_type=None, cohort='Controls', visit=1, rebuild=False):

    rebuild_cases = []

    # build table_dir
    if site is not None:
        lbls_dir = os.path.join(os.getcwd(), "stage_2_training", "ref_mask_dicoms", cohort, site)
    else:
        lbls_dir = os.path.join(os.getcwd(), "stage_2_training", "ref_mask_dicoms", cohort)

    dest_dir_lbls = os.path.join(os.getcwd(), "stage_2_training", "nnunet_raw", "labelsTr")
    os.makedirs(dest_dir_lbls, exist_ok=True)

    if cohort == 'Controls':
        lbls_database = db.series(lbls_dir, contains='kidney_masks')
        lbls_database = [i for i in lbls_database if i[2][0] == f'Visit{visit}']     
    else:
        if study_type is not None:
            lbls_database = db.series(lbls_dir, contains='kidney_masks')
            lbls_database = [i for i in lbls_database if i[2][0] == study_type] 

    for label in lbls_database:
        case_id = label[1]
        if case_id not in ('3128_086', '5128_060', '7128_138'):
            continue

        its_precontrast = check_precontrast_ids(case_id, study_type)

        if cohort == 'Controls':
            output_path = os.path.join(dest_dir_lbls, f"{case_id}_V{visit}.nii.gz")
        else:
            if its_precontrast:
                # this ID belongs to a precontrast image
                if study_type == "Followup":
                    output_path = os.path.join(dest_dir_lbls, f"{case_id}_01.nii.gz")
                else:
                    output_path = os.path.join(dest_dir_lbls, f"{case_id}_00.nii.gz")

            else:
                # this ID belongs to a postcontrast image
                if study_type == "Followup":
                    output_path = os.path.join(dest_dir_lbls, f"{case_id}_1.nii.gz")
                else:
                    output_path = os.path.join(dest_dir_lbls, f"{case_id}.nii.gz")
        
        if os.path.exists(output_path):
            if cohort == 'Controls':
                tqdm.write(f'Case {case_id}_V{visit} exists in folder, skipping!')
            else:
                if its_precontrast:
                    tqdm.write(f'Case {case_id} precontrast {study_type} exists in folder, skipping!')
                else:
                    tqdm.write(f'Case {case_id} postcontrast {study_type} exists in folder, skipping!')
            continue

        if rebuild == True:
            if case_id not in rebuild_cases:
                if cohort == 'Controls':
                    tqdm.write(f'Case {case_id} {visit} not in rebuild list, skipping!')
                else:
                    tqdm.write(f'Case {case_id} {study_type} not in rebuild list, skipping!')
                continue        
        
        


        try:
            label_vol = db.volume(label)
            label_arr = label_vol.values
        except Exception as e:
            print(f'Skipping {case_id}... {e}')
            continue


        affine = label_vol.affine
        nii_label = nib.Nifti1Image(label_arr, affine)
        nib.save(nii_label, output_path)

        print(f'Saved Nifti in folder {output_path}')

        


def prep_img_nnunet():
    visits = [1,2,3,4,5]
    for v in visits:
        prep_tr_data_nnunet(visit=v)

    sites_1 = ['Bari', 'Leeds', 'Sheffield']
    for site in tqdm(sites_1, desc='Sites Completed', unit='site'):
        prep_tr_data_nnunet(site, 'Baseline', 'Patients', precontrast=True)
        prep_tr_data_nnunet(site, 'Baseline', 'Patients')

    sites_2 = ['Bordeaux', 'Exeter', 'Turku']
    for site in tqdm(sites_2, desc='Sites Completed', unit='site'):
        for study in ('Baseline', 'Followup'):
            prep_tr_data_nnunet(site, study, 'Patients', precontrast=True)
            prep_tr_data_nnunet(site, study, 'Patients')

def prep_lbls_nnunet():
    visits = [1,2,3,4,5]
    for v in visits:
        prep_labels(visit=v)
    sites_1 = ['Bari', 'Leeds', 'Sheffield']
    for site in tqdm(sites_1, desc='Sites Completed', unit='site'):
        prep_labels(site, 'Baseline', 'Patients')
    
    sites_2 = ['Bordeaux', 'Exeter', 'Turku']
    for site in tqdm(sites_2, desc='Sites Completed', unit='site'):
        for study in  ('Baseline', 'Followup'):
            prep_labels(site, study, 'Patients')

def run():
    prep_img_nnunet()
    prep_lbls_nnunet()

if __name__ == '__main__':
    run()
    