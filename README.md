# Segmentation Assignment
## Installation
First clone the repository git clone and install all requirements in the virtual environtment
https://github.com/nikhanal/segmentation_with_sam2.git
cd segmentation_with_sam2
pip install pip install -r requirements.txt

SAM 2 needs to be installed first before use. 
git clone https://github.com/facebookresearch/sam2.git && cd sam2

and Download Required checkpoints 
cd checkpoints && \
./download_ckpts.sh && \
cd ..

## Segment Image 
python segment_image.py
