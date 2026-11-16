import os
import shutil
import subprocess
import argparse


def run(build, dataid):

    raw_data = os.path.join(build, 'stage_5_data_prep', 'nnUNet_raw')
    preproc_data = os.path.join(build, 'stage_6_preprocess', 'nnUNet_preprocessed')
    results = os.path.join(build, 'stage_7_results', 'nnUNet_results')

    # Ensure folders exist
    os.makedirs(preproc_data, exist_ok=True)
    os.makedirs(results, exist_ok=True) 

    # Define environment variables
    # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md
    os.environ["nnUNet_raw"] = raw_data
    os.environ["nnUNet_preprocessed"] = preproc_data
    os.environ["nnUNet_results"] = results
    
    
    # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md

    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d",
        dataid,
        "-c", 
        "3d_fullres",
        "-pl",
        "nnUNetPlannerResEncL",
        "--verify_dataset_integrity",
    ]

    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        encoding="utf-8",   # <-- force UTF-8 decoding
        errors="replace"    # <-- avoids crash if weird bytes appear
    )

    # Stream logs in real-time
    for line in process.stdout:
        print(line, end="")

    process.wait()  # wait for completion



if __name__ == '__main__':
    dataid = '001'
    BUILD = os.path.join(os.getcwd(), 'iBEAt_Build', 'ibeatsegmentator')
    run(BUILD, dataid)