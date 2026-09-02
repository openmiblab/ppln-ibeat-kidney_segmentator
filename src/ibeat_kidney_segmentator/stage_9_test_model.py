import os
import shutil
import subprocess
import argparse


def run(build, i, o, f, chk=None):

    # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md

    cmd = [
        "nnUNetv2_predict",
        "-i", i,
        "-o", o,
        "-d", "001",
        "-c", "3d_fullres",
        "-f", f,
        "-chk", chk
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

    BUILD = os.path.join("")
    i = []
    o = []
    f = []
    chk = []

    # Comment for the cluster
    os.environ['nnUNet_n_proc_DA'] = '4' # Set in .sh file
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    run(build=BUILD, i=i, o=o, f=f, chk=chk)