import os
import logging
import argparse
from pathlib import Path

import numpy as np
import dbdicom as db
from tqdm import tqdm
from miblab_plot import mosaic_overlay

from ibeat_kidney_shape.utils import data

MODULE_DIR = Path(__file__).resolve().parent


def run(build):

    dir_data = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data') 
    dir_masks = os.path.join(build, 'kidney_shape', 'stage_1_segment') 
    dir_output = os.path.join(build, 'kidney_shape', 'stage_2_display')
    os.makedirs(dir_output, exist_ok=True)

    logging.basicConfig(
        filename=f"{dir_output}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    for database in ['Controls', 'Patients']:
        db_data = os.path.join(dir_data, database) 
        db_masks = os.path.join(dir_masks, database) 
        db_mosaics = os.path.join(dir_output, database) 
        run_db(db_data, db_masks, db_mosaics)


def run_db(db_data, db_masks, db_mosaics):

    record = data.dixon_record(MODULE_DIR)
    class_map = {1: "kidney_left", 2: "kidney_right"}
    os.makedirs(db_mosaics, exist_ok=True)

    # Loop over the masks
    for mask in tqdm(db.series(db_masks), 'Displaying masks..'):

        # Get the corresponding outphase series
        patient_id, study = mask[1], mask[2][0]
        sequence = data.dixon_series_desc(record, patient_id, study)
        series_op = [db_data, patient_id, study, f'{sequence}_out_phase']
        png_file = os.path.join(db_mosaics, f'{patient_id}_{study}_{sequence}.png')

        # Skip if file exists
        if os.path.exists(png_file):
             continue
        
        try:
            # Load arrays and build ROIs
            op_arr = db.volume(series_op).to_right_handed().values
            mask_arr = db.volume(mask).to_right_handed().values
            rois = {roi: (mask_arr==idx).astype(np.int16) for idx, roi in class_map.items()}

            # Build mosaic and log success
            mosaic_overlay(op_arr, rois, png_file, vmin=0, vmax=np.percentile(op_arr, 90), margin=[16,16,2], opacity=0.75)
            logging.info(f"Success building mosaic for {patient_id}, {study}, {sequence}.")

        except:
            logging.exception(f"Error building mosaic for {patient_id}, {study}, {sequence}.")


if __name__=='__main__':

    BUILD = r'C:\Users\md1spsx\Documents\Data\iBEAt_Build'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)
