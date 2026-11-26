# Object Tracking & Optical Flow Laboratory

OpenCV объект дагах (`BOOSTING`, `MIL`, `KCF`, `TLD`, `MedianFlow`, `CSRT`, `CSRT_REDETECT` + `STATIC` суурь арга) болон optical flow (`Shi–Tomasi + pyramidal Lucas–Kanade`, `Farnebäck dense`, `DISOpticalFlow` — `ULTRAFAST`/`FAST`/`MEDIUM`) алгоритмуудыг сургалтын PDF-ээс давтан ажиллуулж, бодит (`DAVIS 2017`, `Middlebury`, `MPI Sintel`, `LaSOT`) болон CC0 фото-хяналттай өгөгдөл дээр ижил нөхцөлөөр хэмжсэн лабораторийн ажил.

## Орчин ба техник үзүүлэлт (шалгасан)

- **OS:** Linux (`x86_64`)
- **CPU:** Intel(R) Core(TM) 5 120U (benchmark-д `cv2.setNumThreads(1)`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` гэж 1 урсгалд бэхэлсэн; 1 warmup + 5 дахин гүйлтийн медиан ба p95)
- **Python:** `3.14.4`
- **OpenCV:** `opencv-contrib-python==4.14.0.94` (нэг орчинд дан contrib wheel; OpenCV `5.0.0.93`-тай харьцуулж `tracker_bindings_cv500.csv`-д мөн шалгасан)

## Төслийн бүтэц

```text
.
├── config.py                      # PDF-ийн гиперпараметр, 1-thread тохиргоо, замууд
├── metrics.py                     # IoU, EPE, бүсчилсэн EPE, .flo уншигч/бичигч, маск->bbox, сэргэлт
├── trackers.py                    # cv2/cv2.legacy factory, StaticBaselineTracker, CSRTReDetectTracker
├── tracker_binding_check.py       # cv2 ба cv2.legacy binding шалгалт, init() None/bool шалгалт
├── dataset_prep.py                # Middlebury, Wikimedia CC0, DAVIS, MPI Sintel, LaSOT татагч + халхлалтын шат
├── reproduce_pdf.py               # 2 PDF-ийн 3 туршилтын GUI-гүй давталт ба вебкамер smoke run
├── lk_track.py                    # OpenCV samples/python/lk_track.py (Apache-2.0) өөрөө ажилладаг хувилбар
├── opt_flow.py                    # OpenCV samples/python/opt_flow.py (Apache-2.0) тогтмол масштаб + color wheel
├── benchmark_track.py             # Tracker x (үндсэн 5, халхлалтын шат, стресс, ROI мэдрэг чанар, LaSOT/DAVIS)
├── benchmark_flow.py              # Common-subset LK vs Farnebäck vs DIS, бүсчилсэн EPE, цэг ID/drift, камер нөхөлт
├── custom_video_lab.py            # Хэрэглэгчийн утасны 3 видеог тэмдэглэж (--annotate) шалгах (--evaluate) хэрэгсэл
├── make_figure.py                 # 3-frame эхлэл зураг, IoU цагийн шугам, EPE heatmap, камер нөхөлт, CC0 демо MP4
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

Том архивуудыг бүтнээр нь татахгүйгээр ZIP64 HTTP `Range` хүсэлтээр зөвхөн шаардлагатай дарааллуудыг (`DAVIS bmx-trees/car-shadow/scooter-black`, `MPI Sintel clean/alley_1 & final/ambush_5`, `LaSOT bicycle-11`) татаж, `Middlebury` + `Wikimedia CC0` фотогоор хяналттай халхлалтын шатыг бэлдэнэ:

```bash
python dataset_prep.py --all
```

Хэрэв сүлжээгүй орчинд гараар байрлуулах бол:
- `data/wikimedia/Front_Side_of_Bicyle.jpg` — Wikimedia Commons CC0 зураг (`1200x1600`)
- `data/middlebury/other-data/{RubberWhale,Grove2}/frame10.png, frame11.png` болон `data/middlebury/other-gt-flow/{RubberWhale,Grove2}/flow10.flo`
- `data/davis/DAVIS/JPEGImages/480p/{bmx-trees,car-shadow,scooter-black}/*.jpg` болон `Annotations/480p/{...}/*.png`
- `data/sintel/training/{clean/alley_1,final/ambush_5,flow,occlusions,invalid}/...`
- `data/lasot/bicycle-11/{img/*.jpg,groundtruth.txt,full_occlusion.txt,out_of_view.txt}`

## 3. Шалгасан ажиллуулах командууд (GUI-гүй)

```bash
# 1) Tracker binding ба init() буцаалт шалгах
python tracker_binding_check.py

# 2) PDF-ийн бүх туршилтыг GUI-гүйгээр давтаж MP4/PNG/CSV гаргах
python reproduce_pdf.py --mode all-pdf --source data/controlled/bicycle_occ_seq --roi-file results/csv/ctrl_bicycle_occ_gt.csv --no-show

# 3) Тодорхой нэг tracker эсвэл optical flow ажиллуулах (жишээ: CSRT, CSRT_REDETECT, LK)
python reproduce_pdf.py --mode tracking --tracker CSRT --source data/controlled/bicycle_occ_seq --roi 52,160,92,164 --no-show
python reproduce_pdf.py --mode lk --source data/controlled/bicycle_occ_seq --no-show
python reproduce_pdf.py --mode farneback --source data/controlled/bicycle_occ_seq --no-show

# 4) Физик вебкамер (/dev/video0) байвал богино smoke run хийх
python reproduce_pdf.py --mode smoke-webcam

# 5) Бүх tracker болон optical flow benchmark (1 thread, warmup=1, median & p95, algo & end-to-end)
python benchmark_flow.py
python benchmark_track.py

# 6) Өөрийн гар утасны 3 ширээний бичлэгт (тогтвортой, гар хөдөлгөөнтэй, халхлалттай) шошго тавьж шалгах
python custom_video_lab.py --annotate data/phone_videos/stable.mp4 --out-gt results/csv/phone_stable_gt.csv
python custom_video_lab.py --evaluate data/phone_videos/stable.mp4 --gt results/csv/phone_stable_gt.csv

# 7) Зураг, демо MP4, Word тайлан ба unit тестүүд
python make_figure.py
python build_report.py
python -m unittest discover -s tests -v
```

## Лиценз ба дахин нийтлэх хязгаарлалт

- **Програмын код:** Apache License 2.0 (`LICENSE`). OpenCV sample-аас авсан хэсгүүдийг `NOTICE.md`-д эх commit-тай нь бүртгэв.
- **Нийтэлсэн зураг/демо:** `results/figures/` дахь storyboard болон `demo_bicycle_occlusion.mp4` нь **CC0 1.0** (`Front_Side_of_Bicyle.jpg`)-аас үүссэн тул дахин нийтлэх эрх чөлөөтэй.
- **Хасагдсан өгөгдөл:** Сургалтын PDF файлууд, вебкамер/утасны видео, `DAVIS 2017` (`CC BY-NC 4.0`), `Middlebury`, `MPI Sintel`, `LaSOT`-ийн эх файлуудыг `.gitignore`-ээр GitHub-аас хамгаалсан. `Tracking_Lab_Report_MN.docx`-д `DAVIS 2017 bmx-trees`-ийн шинжилгээний зураг эрдэм шинжилгээ, арилжааны бус зорилгоор эшлэлтэй орсон (`CC BY-NC 4.0`).
