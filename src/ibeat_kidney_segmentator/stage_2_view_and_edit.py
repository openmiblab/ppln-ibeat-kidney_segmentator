import os
import numpy as np 
import dbdicom as db 
from tqdm import tqdm 
import napari
from qtpy.QtWidgets import QPushButton
import numpy as np

check_case = []



def edit_case(site=None, group=None, study=None, visit=None, check_cases=False):
    
    if group == 'Controls':
        image_data_dir = os.path.join(os.getcwd(), 'build', 'stage_0_restored_data', 'dixons', group)
        kidney_masks_data_dir = os.path.join(os.getcwd(), 'build', 'stage_1_build_canvas', 'reference_masks', group)
        manual_edits_data_dir = os.path.join(os.getcwd(), 'build', 'stage_1_build_canvas', 'reference_mask_edits', group)        
    else:
        image_data_dir = os.path.join(os.getcwd(), 'build', 'stage_0_restored_data', 'dixons', group, site)
        kidney_masks_data_dir = os.path.join(os.getcwd(), 'build', 'stage_1_build_canvas', 'reference_masks', group, site)
        manual_edits_data_dir = os.path.join(os.getcwd(), 'build', 'stage_1_build_canvas', 'reference_mask_edits', group, site)
    

    database = db.series(image_data_dir, 'dixon_out_phase')
    kidney_database = db.series(kidney_masks_data_dir, 'kidney_masks')

    # for patient baseline or followup
    if group == 'Patients':
        database = [i for i in database if i[2][0] == study]
        kidney_database = [i for i in kidney_database if i[2][0] == study]
    else:
        database = [i for i in database if i[2][0] == f'Visit{visit}']
        kidney_database = [i for i in kidney_database if i[2][0] == f'Visit{visit}']

    for case in database:
        case_id = case[1]
        i_study_or_visit = case[2][0]
        
        if check_cases == True:
            if case_id not in check_case:
                continue

        #select the corresponding mask
        kidney_masks = next(m for m in kidney_database if m[1] == case_id)
        
        k_study_or_visit = kidney_masks[2][0]

        # load volumes
        img_vol = db.volume(case)
        img = img_vol.values
        mask = db.volume(kidney_masks).values

        #load affine
        aff = img_vol.affine


        # view overlays case-by-case  
        viewer = napari.Viewer()
        viewer.add_image(img.T.astype(float), name=f'{case_id}_{i_study_or_visit}_outphase', blending='additive')
        mask_layer = viewer.add_labels(mask.T.astype(np.uint16), name=f'{case_id}_{k_study_or_visit}_mask', opacity=0.5)
        

        def save_mask():
            edited_mask = mask_layer.data

            # transpose back to original orientation
            edited_mask = edited_mask.T
            if group == 'Controls':
                mask_study = [manual_edits_data_dir, case_id, (f'Visit{visit}', 0)]                                    
            else:
                mask_study = [manual_edits_data_dir, case_id, (study, 0)]

            edited_mask_series = mask_study + [(f'kidney_masks', 0)]
            db.write_volume((edited_mask, aff), edited_mask_series, ref=case)

            if group == 'Patients':
                print(f"Saved {case_id} {study} mask to edited folder")
            else:
                print(f"Saved {case_id} V{visit} mask to edited folder")

        save_btn = QPushButton("Save Mask")
        save_btn.clicked.connect(save_mask)

        viewer.window.add_dock_widget(
            save_btn,
            area="right",
            name="Save"
        )

        napari.run()

    print('End of session...Goodbye & Godspeed!')





if __name__ =='__main__':
    Ba = 'Bari'
    B = 'Bordeaux'
    E = 'Exeter'
    L = 'Leeds'
    T = 'Turku'
    S = 'Sheffield'
    C = 'Controls'
    P = 'Patients'
    Base = 'Baseline'
    F = 'Followup'

    edit_case(site=S, group=P, study=Base,  check_cases=True)