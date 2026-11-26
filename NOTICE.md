# Third-Party Code and Dataset License Notices

## 1. Software / Code License (Apache License 2.0)

The laboratory Python source code in this repository is licensed under the Apache License, Version 2.0 (see `LICENSE`).

Portions of `lk_track.py` and `opt_flow.py` are adapted from official OpenCV Python samples:
- `opencv/opencv` `samples/python/lk_track.py` (branch `4.x`, commit `5e80bd3cc922bad5eb1fa02b56bc3db44abb96e9`, Apache License 2.0)
- `opencv/opencv` `samples/python/opt_flow.py` (branch `4.x`, commit `0e40c8a03158e599193d26635fcc13f110c22896`, Apache License 2.0)

Algorithmic descriptions and API verification reference:
- `opencv/opencv_contrib` `modules/tracking` (branch `4.x`, commit `d880a9aa285fa30dc3acbf9f2e959df23d932cca`, Apache License 2.0)

## 2. Dataset Licenses (Separate from Code License)

Raw third-party datasets and course PDFs are **excluded** from this repository (`.gitignore`) and must be obtained directly from their respective upstream sources under their own licenses:

1. **Wikimedia Commons `File:Front_Side_of_Bicyle.jpg`**
   - URL: `https://commons.wikimedia.org/wiki/File:Front_Side_of_Bicyle.jpg`
   - License: **CC0 1.0 Universal Public Domain Dedication**
   - SHA-256: `51c4b3bd6f41eb23c3b4153a019e4b73f25a0cbc013ee535de51aa030bb26266`
   - Published derived demos (`results/figures/*_cc0.png`, `results/figures/demo_bicycle_occlusion.mp4`) derive solely from this CC0 asset and matplotlib plots.

2. **DAVIS 2017 (`bmx-trees`, `car-shadow`, `scooter-black`)**
   - URL: `https://davischallenge.org/davis2017/code.html`
   - License: **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**
   - Citation: Pont-Tuset et al., *The 2017 DAVIS Challenge on Video Object Segmentation*, arXiv:1704.00675, 2017; Perazzi et al., CVPR 2016.
   - Note: `Tracking_Lab_Report_MN.docx` embeds non-commercial failure-analysis figures from `bmx-trees` under CC BY-NC 4.0 with attribution. Do not use for commercial purposes.

3. **Middlebury Optical Flow Evaluation (`RubberWhale`, `Grove2`)**
   - URL: `https://vision.middlebury.edu/flow/data/`
   - Terms: Research and evaluation dataset (Baker et al., *A Database and Evaluation Methodology for Optical Flow*, IJCV 2011). Raw images and `.flo` files are excluded from GitHub.

4. **MPI Sintel Flow Dataset (`training/clean/alley_1`, `training/final/ambush_5`)**
   - URL: `https://sintel.is.tue.mpg.de/`
   - Terms: Academic research and evaluation dataset (Butler et al., *A Naturalistic Open Source Movie for Optical Flow Evaluation*, ECCV 2012). Raw images, `.flo`, and occlusion masks are excluded from GitHub.

5. **LaSOT (`bicycle-11`, frames 1–120)**
   - URL: `http://vision.cs.stonybrook.edu/~lasot/` (mirror used: `https://huggingface.co/datasets/l-lt/LaSOT`)
   - Terms: Academic research dataset (Fan et al., *LaSOT: A High-Quality Benchmark for Large-Scale Single Object Tracking*, CVPR 2019). Raw video frames and ground-truth files are excluded from GitHub.

6. **User-Provided External Video Clips (`stable`, `moving_camera`, `occlusion`)**
   - Local source filenames (excluded from Git via `.gitignore`, hashed for reproducibility):
     - `Тогтвортой камертай жишээ.mp4` (`stable`, `1920x1080`, `23.976 fps`, SHA-256 prefix `77e5daee4220630b`, container metadata `ISO Media file produced by Google Inc.`, clip `orig 664..721`)
     - `Камер өөрөө хөдөлж байгаа жишээ.mp4` (`moving_camera`, `1920x1080`, `25.000 fps`, SHA-256 prefix `df3dd08119b28dae`, container metadata `ISO Media file produced by Google Inc.`, clip `orig 405..515`)
     - `Объект халхлагдаад дахин гарч ирэх жишээ.mp4` (`occlusion`, `720x1280`, `24.000 fps`, SHA-256 prefix `fd1c402e850052c9`, container metadata `ISO Media file produced by Google Inc.`, clip `orig 42..235`)
   - Public redistribution note: Because these `.mp4` files are externally encoded web downloads whose public redistribution license is unverified, neither raw videos nor extracted RGB video frames are committed or published to GitHub. Only local analysis CSVs (`results/csv/phone_*.csv`, `results/csv/track_{stable,moving_camera,occlusion}_*.csv`) and source-image-free coordinate schematics, pure optical-flow HSV/vector plots, and quantitative charts (`results/figures/custom_videos/*.png`) are published; raw RGB QA overlays remain strictly in `results/local_only/`.
