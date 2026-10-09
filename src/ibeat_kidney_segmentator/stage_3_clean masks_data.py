import os 
from tqdm import tqdm 
import dbdicom as db
import re

"""
Clean up mask data from viewer/GUI saved directories
i.e., if segmented masks were generated on wezel, the following
    script re-saves mask in the dedicated harmonised dicom folders
"""

def list_ids_through_data_dir(site, group, study):
    
    if group == 'Controls':
        data_dir = os.path.join(os.getcwd(), 'build', 'stage_1_build_canvas', 'reference_masks', group)
    else:
        data_dir = os.path.join(os.getcwd(), 'build', 'stage_1_build_canvas', 'reference_masks', group, site)
    
    database = db.series(data_dir, 'kidney_masks')
    for case in database:
        case_id = case[1]
        if group != 'Controls':
            if study is not None:
                if case[2][0] == study:
                    print(case_id)
        else:
            study = case[2][0]
            print(case_id)

def list_ids_through_filenames(group):
    
    png_dir = os.path.join(os.getcwd(), 'build', 'stage_1_build_canvas', 'displays', group)
    nii_dir = os.path.join(os.getcwd(), 'training', 'imagesTr')

    for f in os.listdir(png_dir):  
        parts = f.split('_')
        print(f'{parts[0]}_{parts[1]}')  


def write_masks_into_dir(build_path, group, site, study, check_case=False, build_new_dir=False):


    maskpath = os.path.join(build_path,  'stage_1_build_canvas', 'raw_masks') 
    destpath = os.path.join(build_path,  'stage_2_kidney_masks')
    os.makedirs(destpath, exist_ok=True)

    if group == 'Controls':
        sitemaskpath = os.path.join(maskpath, group)
        sitedestpath = os.path.join(destpath, group)
    else: 
        sitemaskpath = os.path.join(maskpath, group, site)
        sitedestpath = os.path.join(destpath, group, site)

    os.makedirs(sitemaskpath, exist_ok=True)


    # Get out phase series
    series_from = db.series(sitemaskpath)
    series_to = db.series(sitedestpath)
    
    
    series_lk = [s for s in series_from if s[3][0]=='LK']
    series_rk = [s for s in series_from if s[3][0]=='RK']
    
    #paitent but dir ok
    #skip filter for special_build cases if they have wrong study entry 
    if not build_new_dir and group != 'Controls':
        series_lk = [s for s in series_lk if s[2][0]== study]
        series_rk = [s for s in series_rk if s[2][0]== study]
   
    #series naming and check/skip if it already exists  
    for case in tqdm(series_lk, desc='Writing LK masks to folder', unit='case'):

        case_id = case[1]
        m_study = case[2][0]
        


        if build_new_dir == True:
            tqdm.write(f'Sending case {case_id} to build new dir...')
            m = re.findall(r'\d+', case_id)
            if not m:
                case_id = case[1]
                print(f'Using default case id: {case_id}')
                m_study = case[2][0]

            else:
                digits = "".join(m)
                if len(digits) >= 7:
                    case_id = f"{digits[:4]}_{digits[4:7]}"
                else:
                    case_id = digits

                m_study = study
                

        if check_case == True:
            if case_id not in fix_cases:
                tqdm.write(f'case {case_id} LK not in fix cases, skipping!')
                continue
        
        if group == 'Patients':
            if m_study != study:
                tqdm.write(f'mask study = {m_study} and does not match with database enqiry = {study}, skipping!')
                continue

        tqdm.write(f'Processing case {case_id} LK {m_study}')
        
        try:
            vol = db.volume(case)
        except Exception as e:
            print(f'skipping case {case_id} {e}')
        database = [sitedestpath, case_id, (m_study, 0)]
        lk_clean = database + [("LK", 0)]
        if lk_clean in series_to:
            print('lk exists in folder, skipping!')
            continue
        try:
            db.write_volume(vol, lk_clean, ref=case)
        except Exception as e:
            print(f'Skipping case {case_id}: {e}')
            continue


    for case in tqdm(series_rk, desc='Writing RK masks to folder', unit='case'):

        case_id = case[1]
        m_study = case[2][0]


        if build_new_dir == True:
            tqdm.write(f'Sending case {case_id} to build new dir...')
            m = re.findall(r'\d+', case_id)
            if not m:
                case_id = case[1]
                print(f'Using default case id: {case_id}')
                m_study = case[2][0]

            else:
                digits = "".join(m)
                if len(digits) >= 7:
                    case_id = f"{digits[:4]}_{digits[4:7]}"
                else:
                    case_id = digits
                    
                m_study = study

        
        if check_case == True:
            if case_id not in fix_cases:
                tqdm.write(f'case {case_id} RK not in fix cases, skipping!')
                continue
        
        if group == 'Patients':
            if m_study != study:
                tqdm.write(f'mask study = {m_study} and does not match with database enqiry = {study}, skipping!')
                continue

        tqdm.write(f'Processing case {case_id} RK {m_study}')
        try:
            vol = db.volume(case)
        except Exception as e:
            print(f'skipping case {case_id} {e}')

        database = [sitedestpath, case_id, (m_study, 0)]
        rk_clean = database + [("RK", 0)]
        if rk_clean in series_to:
            print('rk exists in folder, skipping!')
            continue
        try:
            db.write_volume(vol, rk_clean, ref=case)
        except Exception as e:
            print(f'Skipping case {case_id}: {e}')
            continue


if __name__ == '___main___':
    fix_cases = []
    write_masks_into_dir()