"""
Use the following HPC command to restore dixon clean data 
from the archive directory into the desired folder 
for segmentation data preparation.

----------
From dir use: 
LOCAL_DIR="/shared/abdominal_imaging/Archive/iBEAt_Build/dixon/stage_5_clean_dixon_data"

To dir use: 
REMOTE_DIR="/shared/abdominal_imaging/Shared/Ajo/iBEAt_Build/kidney_segmentation"

rsync -av --progress --no-group --no-perms "$LOCAL_DIR" "$REMOTE_DIR"
-----------



"""