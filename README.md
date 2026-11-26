# Object Tracking & Optical Flow Laboratory

OpenCV объект дагах (`BOOSTING`, `MIL`, `KCF`, `TLD`, `MedianFlow`, `CSRT` + `STATIC` суурь арга) болон optical flow (`Shi–Tomasi + pyramidal Lucas–Kanade`, `Farnebäck dense`) алгоритмуудыг сургалтын PDF-ээс давтан ажиллуулж, бодит (DAVIS 2017, Middlebury) болон CC0 фото-хяналттай өгөгдөл дээр ижил нөхцөлөөр хэмжсэн лабораторийн ажил.

## Орчин ба техник үзүүлэлт (шалгасан)

- **OS:** Linux (`x86_64`)
- **CPU:** Intel(R) Core(TM) 5 120U (benchmark-д `cv2.setNumThreads(1)`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` гэж 1 урсгалд бэхэлсэн)
- **Python:** `3.14.4`
- **OpenCV:** `opencv-contrib-python==4.14.0.94` (нэг орчинд дан contrib wheel; OpenCV `5.0.0.93`-тай харьцуулж `tracker_bindings_cv500.csv`-д мөн шалгасан)

## Төслийн бүтэц

```text
.
├── config.py                      # PDF-ийн гиперпараметр, 1-thread тохиргоо, замууд
├── metrics.py                     # IoU, EPE, .flo уншигч/бичигч, маск->bbox
├── tracker_binding_check.py       # cv2 ба cv2.legacy binding шалгалт, init() None/bool шалгалт
├── dataset_prep.py                # Middlebury, Wikimedia CC0, DAVIS HTTP-Range татагч + хяналттай дараалал
├── reproduce_pdf.py               # 2 PDF-ийн 3 туршилтын GUI-гүй давталт ба вебкамер smoke run
├── lk_track.py                    # OpenCV samples/python/lk_track.py (Apache-2.0) өөрөө ажилладаг хувилбар
├── opt_flow.py                    # OpenCV samples/python/opt_flow.py (Apache-2.0) тогтмол масштаб + color wheel
├── benchmark_track.py             # 7 tracker x 5 дараалал (3 давталтын median ms/frame, IoU, сэргэлт)
├── benchmark_flow.py              # LK vs Farnebäck EPE, цэг алдалт, FB-check ба winSize ablation
├── make_figure.py                 # Storyboard, график, CC0 демо MP4 үүсгэгч
├── build_report.py                # CSV-ээс шууд Tracking_Lab_Report_MN.docx угсрагч
├── tests/test_metrics_and_pipeline.py
├── results/
│   ├── csv/                       # Бүх хэмжилтийн CSV ба dataset_manifest.csv
│   └── figures/                   # Нийтлэх эрхтэй CC0 storyboard, график, демо MP4
├── Tracking_Lab_Report_MN.docx    # Монгол тайлан (энгийн хар бичвэр, зураг, хүснэгт)
├── requirements.txt
├── LICENSE                        # Кодын Apache-2.0 лиценз
└── NOTICE.md                      # Код ба датасетийн тусдаа лицензийн мэдэгдэл
```

## 1. Цэвэр орчин үүсгэж суулгах

Нэг орчинд `opencv-python` ба `opencv-contrib-python`-ийг давхар суулгахгүй, зөвхөн `opencv-contrib-python`-ийг суулгана:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Өгөгдөл татах ба хяналттай туршилт бэлдэх

Том `DAVIS-2017-trainval-480p.zip` (832 MB) архивыг бүтнээр нь татахгүйгээр HTTP `Range` хүсэлтээр зөвхөн `bmx-trees` ба `car-shadow`-г татаж, Middlebury + Wikimedia CC0 фотог бэлдэнэ:

```bash
python dataset_prep.py --all
```

Хэрэв сүлжээгүй орчинд гараар байрлуулах бол:
- `data/wikimedia/Front_Side_of_Bicyle.jpg` — Wikimedia Commons CC0 зураг (`1200x1600`)
- `data/middlebury/other-data/{RubberWhale,Grove2}/frame10.png, frame11.png` болон `data/middlebury/other-gt-flow/{RubberWhale,Grove2}/flow10.flo`
- `data/davis/DAVIS/JPEGImages/480p/{bmx-trees,car-shadow}/*.jpg` болон `data/davis/DAVIS/Annotations/480p/{bmx-trees,car-shadow}/*.png`

## 3. Шалгасан ажиллуулах командууд (GUI-гүй)

```bash
# 1) Tracker binding ба init() буцаалт шалгах
python tracker_binding_check.py

# 2) PDF-ийн бүх туршилтыг GUI-гүйгээр давтаж MP4/PNG/CSV гаргах
python reproduce_pdf.py --mode all-pdf --source data/controlled/bicycle_occ_seq --roi-file results/csv/ctrl_bicycle_occ_gt.csv --no-show

# 3) Тодорхой нэг tracker эсвэл optical flow ажиллуулах (жишээ: CSRT ба LK)
python reproduce_pdf.py --mode tracking --tracker CSRT --source data/controlled/bicycle_occ_seq --roi 52,160,92,164 --no-show
python reproduce_pdf.py --mode lk --source data/controlled/bicycle_occ_seq --no-show
python reproduce_pdf.py --mode farneback --source data/controlled/bicycle_occ_seq --no-show

# 4) Физик вебкамер (/dev/video0) байвал богино smoke run хийх
python reproduce_pdf.py --mode smoke-webcam

# 5) Бүх tracker болон optical flow benchmark (1 thread, 3 давталтын медиан)
python benchmark_track.py
python benchmark_flow.py

# 6) Зураг, демо MP4, Word тайлан ба unit тестүүд
python make_figure.py
python build_report.py
python -m unittest discover -s tests -v
```

## Лиценз ба дахин нийтлэх хязгаарлалт

- **Програмын код:** Apache License 2.0 (`LICENSE`). OpenCV sample-аас авсан хэсгүүдийг `NOTICE.md`-д эх commit-тай нь бүртгэв.
- **Нийтэлсэн зураг/демо:** `results/figures/` дахь storyboard болон `demo_bicycle_occlusion.mp4` нь **CC0 1.0** (`Front_Side_of_Bicyle.jpg`)-аас үүссэн тул дахин нийтлэх эрх чөлөөтэй.
- **Хасагдсан өгөгдөл:** Сургалтын PDF файлууд, вебкамерын зураг, `DAVIS 2017` (`CC BY-NC 4.0`), `Middlebury` эх файлуудыг `.gitignore`-ээр GitHub-аас хамгаалсан. `Tracking_Lab_Report_MN.docx`-д `DAVIS 2017 bmx-trees`-ийн нэг эвдрэлийн зураг эрдэм шинжилгээ, арилжааны бус тайланд эшлэлтэй орсон (`CC BY-NC 4.0`).
