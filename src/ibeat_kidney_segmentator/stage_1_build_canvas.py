"""
Functions that were used to generate the canvas for semi-automated 
definition of the training data. This functions are preserved for 
future reference but not currently used as part of the pipeline.

They will be needed when new training data are to be generated from 
scratch in the future.
"""

import os
import logging
import numpy as np
import dbdicom as db
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import napari
import utils.data
from tqdm import tqdm
import re

fix_cases = []

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
 
def rewrite_mask_into_dir(build_path, group, site, study, check_case=False, build_new_dir=False):


    maskpath = os.path.join(build_path,  'stage_1_build_canvas', 'raw_masks') 
    destpath = os.path.join(build_path,  'stage_1_build_canvas', 'clean_masks')
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


def compute_canvas(build_path, group, site=None):


    datapath = os.path.join(build_path, 'stage_0_restored_data', 'dixons') 
    maskpath = os.path.join(build_path, 'stage_1_canvas') 
    os.makedirs(maskpath, exist_ok=True)

    if group == 'Controls':
        sitedatapath = os.path.join(datapath, group) 
        sitemaskpath = os.path.join(maskpath, group)
    else:
        sitedatapath = os.path.join(datapath, group, site) 
        sitemaskpath = os.path.join(maskpath, group, site)
    os.makedirs(sitemaskpath, exist_ok=True)


    # List of selected dixon series
    record = utils.data.dixon_record(parent='ibeat_kidney_segmentator')

    # Get out phase series
    series = db.series(sitedatapath)
    series_out_phase = [s for s in series if s[3][0][-9:]=='out_phase']

    # Loop over the out-phase series
    for series_op in tqdm(series_out_phase, desc='Computing canvas', unit='case'):

        # Patient and output study
        patient = series_op[1]
        study = series_op[2][0]
        series_op_desc = series_op[3][0]
        sequence = series_op_desc[:-10]

        # Skip if it is not the right sequence
        selected_sequence = utils.data.dixon_series_desc(record, patient, study)
        if sequence != selected_sequence:
            continue

        # Skip if the kidney masks already exist
        mask_study = [sitemaskpath, patient, (study,0)]
        mask_series_0 = mask_study + [(f'kidney_canvas_0', 0)]
        if mask_series_0 in db.series(mask_study):
            continue

        # Get fat image
        series_fi = series_op[:3] + [(sequence + '_fat', 0)]
        if series_fi not in series:
            continue

        # Read the out-phase and water volumes
        try:
            op = db.volume(series_op)
            fi = db.volume(series_fi)
        except Exception as e:
            logging.error(f"Patient {patient} - error reading I-O {sequence}: {e}")
            continue

        try:
            canvas_maps = sequential_kmeans([fi.values, op.values])
        except Exception as e:
            logging.error(f"Error computing canvas for {patient} {sequence}: {e}")
            continue
        
        for i, c in enumerate(canvas_maps):
            mask_series = mask_study + [f'kidney_canvas_{i}']
            db.write_volume((c, op.affine), mask_series, ref=series_op)

        # Include source images for reference
        db.write_volume(op, mask_study + [f'dixon_out_phase'], ref=series_op)
        db.write_volume(fi, mask_study + [f'dixon_fat'], ref=series_fi)


def display_canvas(build_path, group, site=None):

    maskpath = os.path.join(build_path, 'stage_1_canvas') 
    if group == 'Controls':
        sitemaskpath = os.path.join(maskpath, group)
    else:
        sitemaskpath = os.path.join(maskpath, group, site)

    # Get out phase series
    studies = db.studies(sitemaskpath)

    # Loop over the studies
    for study in studies:

        arrays = []
        names = []
        for series in db.series(study):
            arr = db.volume(series).values
            arrays.append(arr.T)
            names.append(series[-1][0])

        show_arrays_in_napari(arrays, names=names)



def sequential_kmeans(features, n_clusters=2):

    f = features[0]
    clusters = kmeans([f], n_clusters=n_clusters)
    for f in features[1:]:
        cf = []
        for mask in clusters:
            cf += kmeans([f], mask=mask, n_clusters=n_clusters)
        clusters = cf

    return clusters


def kmeans(features, mask=None, n_clusters=2, normalize=True):

    shape = features[0].shape

    # extract the indices of all pixels under the mask
    if mask is not None:
        mask_array = np.ravel(mask)
        mask_indices = tuple(mask_array.nonzero())

    # Create array with shape (n_samples, n_features) and mask if needed.
    array = []
    for arr in features:
        arr = np.ravel(arr)
        if mask is not None:
            arr = arr[mask_indices]
        array.append(arr)
    array = np.vstack(array).T

    # Normalize
    if normalize:
        array = StandardScaler().fit_transform(array)
    
    # Perform the K-Means clustering.
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=3, verbose=1).fit(array)

    # Create an output array for the labels
    if mask is None:
        output_array = 1 + kmeans.labels_
    else:
        output_array = np.zeros(shape).ravel()
        output_array[mask_indices] = 1 + kmeans.labels_ 
    output_array = output_array.reshape(shape)

    # Save each cluster as a separate mask
    clusters = []
    for cluster in range(1, 1 + n_clusters):
        array_cluster = np.zeros(shape)
        array_cluster[output_array == cluster] = 1
        clusters.append(array_cluster)

    return clusters




def show_arrays_in_napari(arrays, names=None, contrast_limits=None):
    """
    Opens a Napari viewer and displays multiple NumPy arrays as image layers.
    """
    if len(arrays) == 0:
        raise ValueError("You must provide at least one array.")

    arrays = list(arrays)
    shapes = [a.shape[:2] for a in arrays]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(f"All arrays must share the same spatial shape; found shapes: {shapes}")

    n = len(arrays)
    if names is None:
        names = [f"layer_{i}" for i in range(n)]
    if contrast_limits is None:
        contrast_limits = [[a.min(), a.max()] for a in arrays]

    viewer = napari.Viewer()
    for i, arr in enumerate(arrays):
        viewer.add_image(
            arr,
            name=names[i],
            contrast_limits=contrast_limits[i],
        )

    napari.run()
    return viewer

def run(build):

    compute_canvas(build, 'Controls')

    sites_1 = ['Bari', 'Leeds', 'Sheffield']
    for site in tqdm(sites_1, desc='Sites Completed', unit='site'):
        compute_canvas(build, 'Patients', site)

    sites_2 = ['Bordeaux', 'Exeter', 'Turku']
    for site in tqdm(sites_2, desc='Sites Completed', unit='site'):
        for study in ('Baseline', 'Followup'):
            compute_canvas(build, 'Patients', site, study)

    # #Once mask is generated in gui, use the following function to rebuild directory
    # group=[], site=[], study=[],
    # rewrite_mask_into_dir(build_path, group, site, study, build_new_dir=False)

    # #List IDs for datakeeping purposes:
    # list_ids_through_data_dir()
    # list_ids_through_filenames()


if __name__ == '__main__':
    run()

