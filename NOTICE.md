# Third-Party Code and Dataset License Notices

## 1. Software / Code License (Apache License 2.0)

The laboratory Python source code in this repository is licensed under the Apache License, Version 2.0 (see `LICENSE`).

Portions of `lk_track.py` and `opt_flow.py` are adapted from official OpenCV Python samples:
- `opencv/opencv` `samples/python/lk_track.py` (branch `4.x`, commit `5e80bd3cc922bad5eb1fa02b56bc3db44abb96e9`, Apache License 2.0)
- `opencv/opencv` `samples/python/opt_flow.py` (branch `4.x`, commit `0e40c8a03158e599193d26635fcc13f110c22896`, Apache License 2.0)

Algorithmic descriptions and API verification reference:
- `opencv/opencv_contrib` `modules/tracking` (branch `4.x`, commit `d880a9aa285fa30dc3acbf9f2e959df23d932cca`, Apache License 2.0)

## 2. Dataset Licenses (Separate from Code License)

Raw third-party datasets and course PDFs are **excluded** via `.gitignore`. The three video inputs listed in item 6 are used in this private repository.

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
   - `Тогтвортой камертай жишээ.mp4` → `data/phone_videos/stable.mp4`. Subaru EyeSight miniature commercial clip, `1920 × 1080`, `24000/1001 fps`, 2,328 frames, 97.097 s. SHA-256 `77e5daee4220630b8bd9b4d2c9860eb7678530b33e848adf6087b6dd2a996c95`. Measured source frames 664–721.
   - `Камер өөрөө хөдөлж байгаа жишээ.mp4` → `data/phone_videos/moving_camera.mp4`. Insta360 ONE X2 RC buggy and dog clip. Source and repository copy are both `1920 × 1080`, 25 fps, 2,299 frames, 91.960 s. Source SHA-256 `df3dd08119b28dae02fa80675eb5240665d1ce1dea035b15a24cadbfab58f71e`. Repository copy SHA-256 `188c475d8d540a7cc78b447e0349bdbba46bf38b0d403d8cc7acd2ed2716edd0`. Measured source frames 405–515.
   - `Объект халхлагдаад дахин гарч ирэх жишээ.mp4` → `data/phone_videos/occlusion.mp4`. Invisibility Shield lenticular refraction apple clip, `720 × 1280`, 24 fps, 241 frames, 10.042 s. SHA-256 `fd1c402e850052c9f446acf846c39bf816e7f2c17561c308e9f71fa54566294f`. Measured source frames 42–235.
   - The user-provided DASH inputs carry the container tag `Google Inc.`. The `stable.mp4` and `occlusion.mp4` copies retain the source bytes. The complete `moving_camera.mp4` was re-encoded directly from the source with two-pass H.264 at 2,110 kb/s (52,145,895 → 24,362,410 bytes) after the GitHub blob upload rejected the larger file. All 2,299 frame timestamps match the source. The derived `results/figures/custom_videos/tracking_{stable,moving_camera,occlusion}.gif` files show selected frames from the measured ranges, cropped and annotated with CSV bounding boxes.
   - The source videos and GIFs are included for this lab in the private repository. Public redistribution terms remain unverified. The code license does not cover these clips.

7. **Wikimedia Commons `File:Arrivée en gare du Léman Express pour Coppet (2019-12-16).webm`**
   - Page URL: `https://commons.wikimedia.org/wiki/File:Arriv%C3%A9e_en_gare_du_L%C3%A9man_Express_pour_Coppet_(2019-12-16).webm`
   - Direct Media URL: `https://upload.wikimedia.org/wikipedia/commons/9/9b/Arriv%C3%A9e_en_gare_du_L%C3%A9man_Express_pour_Coppet_%282019-12-16%29.webm`
   - Author / Credit: `Thibaut120094` (Own work, recorded 2019-12-16)
   - License: **CC0 1.0 Universal Public Domain Dedication** (`https://creativecommons.org/publicdomain/zero/1.0/deed.en`)
   - SHA-1 (upstream): `954c86af970dff484f6f72218778fc147fec794f`
   - Modifications: Sliced continuous range `orig #110–#260` (`151 frames`, `0.00–5.00 s @ 30.000 fps`), cropped identically (`[x=330..920, y=125..520]`) for side-by-side display, overlaid manually verified GT and `KCF`/`CSRT` tracker boxes, and encoded at real-time 1.0× speed (`10 fps`, 51 frames) into `results/figures/custom_videos/tracking_demo.gif`, `tracking_demo_keyframes.png`, and `summary_good_vs_fail.png` (left panel). Raw `.webm` remains in `data/commons/` (excluded via `.gitignore`).

8. **Wikimedia Commons Squirrel Full-Occlusion Sequence (`squirrel_occlusion`, CC0 1.0 controlled sequence from real photos)**
   - Squirrel source: `File:Squirrel eating nut in grass.png` (`https://commons.wikimedia.org/wiki/File:Squirrel_eating_nut_in_grass.png`), Author `TylerMascola` (Own work), **CC0 1.0 Universal Public Domain Dedication** (`https://creativecommons.org/publicdomain/zero/1.0/deed.en`), upstream original SHA-1 `9de60ec36c67fe8fb8d2ce21b3c38c10bb1de83e` (`3840px` preview copy cached under `data/wikimedia/`).
   - Tree trunk & park background source: `File:Squirrel on tree trunk.jpg` (`https://commons.wikimedia.org/wiki/File:Squirrel_on_tree_trunk.jpg`), Author `Ilikecheese` (Own work), **CC0 1.0 Universal Public Domain Dedication**, raw resolution `3240×4320`, upstream original SHA-1 `6bd0347b5b430deb103ce6fee3584ae92b93284e`.
   - Modifications & Transparency Note: Nine Wikimedia Commons real squirrel videos (including `Morton Arboretum Winter Squirrel 2005.webm`, `Squirrels running up the wall.webm`, etc.) were visually audited first and found not to contain a continuous single-squirrel complete `0%` trunk occlusion & opposite-side reappearance. Therefore `squirrel_occlusion_lab.py` constructs a `720×420` controlled 66-frame photo sequence (`12 fps`, `5.50 s`, `#0–#65`, frames `#28–#36` = 9 frames labeled absent using fewer than 25 visible mask pixels; `#28` retains a visible fraction of 0.05% and `#29–#36` have no visible mask pixels; `reappear_frame = #37`), isolating a clean GrabCut foreground squirrel matte over squirrel-free rotated park lawn/path regions with a foreground tree trunk mask in the center. Derived outputs (`results/figures/custom_videos/squirrel_occlusion.gif`, `squirrel_occlusion_keyframes.png`, and `summary_good_vs_fail.png` right panel) explicitly label this sequence as *"Бодит фото ашигласан хяналттай дараалал"*. Raw CC0 photos and generated PNG frames stay under `data/` (excluded via `.gitignore`).
