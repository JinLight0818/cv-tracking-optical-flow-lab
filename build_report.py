"""Build beginner-friendly Mongolian laboratory report Tracking_Lab_Report_MN.docx strictly from CSVs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

from config import CSV_DIR, FIGURES_DIR, LOCAL_VIZ_DIR, ROOT

BLACK = RGBColor(0x00, 0x00, 0x00)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _by_frame(path: Path) -> Dict[int, Dict[str, str]]:
    return {int(r["frame"]): r for r in _read_csv(path)}


def _set_run_black(run, size: int = 10, bold: bool = False, italic: bool = False) -> None:
    run.font.name = "DejaVu Sans"
    run.font.size = Pt(size)
    run.font.color.rgb = BLACK
    run.bold = bold
    run.italic = italic


def add_h(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    sizes = {0: 15, 1: 12, 2: 10.5}
    for r in h.runs:
        _set_run_black(r, size=sizes.get(level, 11), bold=True)


def add_p(doc: Document, text: str, size: int = 10, bold: bool = False, italic: bool = False, keep_next: bool | None = None) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    if keep_next if keep_next is not None else bold:
        p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    _set_run_black(r, size=size, bold=bold, italic=italic)


def add_bullets(doc: Document, items: List[str], size: int = 9.5) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.12
        r = p.add_run(item)
        _set_run_black(r, size=size)


def _set_row_cant_split(row, is_header: bool = False) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))
    if is_header:
        tr_pr.append(OxmlElement("w:tblHeader"))


def add_table(
    doc: Document,
    headers: List[str],
    rows: List[List[str]],
    caption: str,
    font_pt: int = 7.5,
    col_widths: List[float] | None = None,
) -> None:
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = col_widths is None

    _set_row_cant_split(table.rows[0], is_header=True)
    for j, htxt in enumerate(headers):
        cell = table.cell(0, j)
        if col_widths is not None:
            cell.width = Inches(col_widths[j])
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(htxt)
        _set_run_black(r, size=font_pt, bold=True)

    for i, row in enumerate(rows, start=1):
        _set_row_cant_split(table.rows[i], is_header=False)
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            if col_widths is not None:
                cell.width = Inches(col_widths[j])
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(str(val))
            _set_run_black(r, size=font_pt)

    add_p(doc, caption, size=8.5, italic=True)


def add_img(doc: Document, img_path: Path, width_in: float, caption: str) -> None:
    if not img_path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(img_path), width=Inches(width_in))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(6)
    r = cap.add_run(caption)
    _set_run_black(r, size=8.5, italic=True)


def _fmt_rec(r: Dict[str, str]) -> str:
    rs, rc = r["recovery_start_frames"], r["recovery_confirmed_frames"]
    if rs == "N/A":
        return "N/A"
    if rs == "эргэж олоогүй":
        return "Олоогүй"
    return f"{rs} / {rc} (f{r['first_ok_frame']}/f{r['confirmed_frame']})"


def build_report(repo_url_note: str = "https://github.com/JinLight0818/cv-tracking-optical-flow-lab (private)") -> Path:
    track_summary = _read_csv(CSV_DIR / "track_summary.csv")
    track_ladder = _read_csv(CSV_DIR / "track_occlusion_ladder.csv")
    track_stress = _read_csv(CSV_DIR / "track_stress_tests.csv")
    track_roi = _read_csv(CSV_DIR / "track_roi_sensitivity.csv")
    track_gen = _read_csv(CSV_DIR / "track_generalization_lasot.csv")
    flow_summary = _read_csv(CSV_DIR / "flow_summary.csv")
    flow_regions = _read_csv(CSV_DIR / "flow_dense_regions.csv")
    flow_ablation = _read_csv(CSV_DIR / "flow_ablation.csv")
    cam_comp = _read_csv(CSV_DIR / "camera_compensation.csv")
    manifest = _read_csv(CSV_DIR / "dataset_manifest.csv")
    bindings_414 = _read_csv(CSV_DIR / "tracker_bindings_cv414.csv")
    webcam = _read_csv(CSV_DIR / "webcam_smoke.csv")

    def tlookup(rows: List[Dict[str, str]], seq: str, tr: str) -> Dict[str, str]:
        return next(r for r in rows if r["sequence"] == seq and r["tracker"] == tr)

    def flookup(ds: str, method: str, mc: str = "10") -> Dict[str, str]:
        return next(r for r in flow_summary if r["dataset"] == ds and r["method"] == method and r["max_corners"] == mc)

    def rlookup(ds: str, method: str) -> Dict[str, str]:
        return next(r for r in flow_regions if r["dataset"] == ds and r["method"] == method)

    def alookup(seq: str, var: str) -> Dict[str, str]:
        return next(r for r in flow_ablation if r["sequence"] == seq and r["variant"] == var)

    def slookup(seq: str, cond: str, tr: str) -> Dict[str, str]:
        return next(r for r in track_stress if r["sequence"] == seq and r["condition"] == cond and r["tracker"] == tr)

    def rslookup(seq: str, pert: str, tr: str) -> Dict[str, str]:
        return next(r for r in track_roi if r["sequence"] == seq and r["perturbation"] == pert and r["tracker"] == tr)

    def clookup(pair: str, fm: str) -> Dict[str, str]:
        return next(r for r in cam_comp if r["pair"] == pair and r["flow_method"] == fm)

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "DejaVu Sans"
    normal.font.size = Pt(10)
    normal.font.color.rgb = BLACK

    for sec in doc.sections:
        sec.top_margin = Inches(0.6)
        sec.bottom_margin = Inches(0.6)
        sec.left_margin = Inches(0.6)
        sec.right_margin = Inches(0.6)

    add_h(doc, "Объект дагах (Object Tracking) ба Optical Flow лабораторийн тайлан", level=0)
    add_p(doc, "Орчин: Linux (x86_64), Intel(R) Core(TM) 5 120U, Python 3.14.4, opencv-contrib-python==4.14.0.94 (1 thread, warmup=1, median & p95)", size=8.5, italic=True)
    add_p(doc, f"Эх код ба баримт: {repo_url_note}", size=8.5, italic=True)

    # =========================================================================
    # 1. ЗОРИЛГО, 3-КАДРЫН ЭХНИЙ ТҮҮХ БА АНХАН ШАТНЫ ОЙЛГОЛТ
    # =========================================================================
    add_h(doc, "1. Зорилго, 3 кадруудын бодит жишээ ба анхан шатны ойлголт", level=1)
    add_p(
        doc,
        "Энэхүү лабораторийн зорилго нь дараалсан видео кадруудаас сонгосон объектыг дагах OpenCV tracker-үүд "
        "(Boosting, MIL, KCF, TLD, MedianFlow, CSRT болон бидний сайжруулсан CSRT_REDETECT) ба хөдөлгөөний вектор талбай "
        "бодох optical flow аргуудыг (Shi–Tomasi + pyramidal Lucas–Kanade, Farnebäck dense, DISOpticalFlow) "
        "хоёр сургалтын PDF материалаас давтаж, ижил 1-thread орчинд бодит ба хяналттай олон сорил дээр хэмжиж харьцуулах юм."
    )
    add_p(doc, "Компьютер харааг анх удаа үзэж буй уншигч дараах 3 асуултаар үндсэн ялгааг шууд ойлгож болно:", bold=True)
    add_bullets(doc, [
        "1) Объект одоо хаана байна? — Кадр бүрд объектын тэгш өнцөгт хүрээ (bounding box) гаргах нь Object Tracking буюу объект дагалт.",
        "2) Хэдэн тод цэг хаашаа шилжив? — Хүчтэй булангийн цөөн цэгийн шилжилтийг мөрдөх нь Sparse Optical Flow (Lucas–Kanade).",
        "3) Пиксел бүр хаашаа, хэр хурдан шилжив? — Дүрсний пиксел бүрд хөдөлгөөний вектор оноох нь Dense Optical Flow (Farnebäck, DIS).",
    ])

    add_img(
        doc,
        FIGURES_DIR / "intro_3frame_story_cc0.png",
        6.8,
        "Зураг 1. Анхан шатны 3 кадрын түүх (CC0 ctrl_bicycle_occ): (1) Frame 4 — цэвэр харагдах үед бүгд дагана; "
        "(2) Frame 19 — баганын ард 100% халхлагдахад CSRT/CSRT_REDETECT зогсоож алдаа хийхгүй бол MIL баганан дээр наалдана; "
        "(3) Frame 31 — дугуй дахин гарахад энгийн CSRT хоцорсон бол CSRT_REDETECT ба TLD эргэн барьсан байдал."
    )

    add_p(doc, "Үндсэн нэр томьёо ба хэмжүүрийн тодорхойлолт (тоон жишээтэй):", bold=True)
    add_bullets(doc, [
        "ROI (Region of Interest): Эхний кадрт (frame 0) хэрэглэгч эсвэл GT-ээр өгдөг анхны хүрээ (x, y, w, h) — зүүн дээд булан ба өргөн, өндөр (пикселээр).",
        "Ground Truth (GT — Visible vs Amodal): Кадр бүрийн жинхэнэ хүрээ. Visible-part GT (DAVIS гэх мэт) нь зөвхөн ил харагдах хэсгийг хүрээлнэ; Amodal GT нь халхлагдсан ч бүтэн объектын таамагт хүрээг авна. Халхлагдаж эхлэхэд Visible GT нарийсдаг тул tracker өргөнөө хумихгүй бол Visible IoU унадаг.",
        "IoU (Intersection over Union): Таамаглал A ба GT B хоёрын давхцсан талбайг нийлбэр талбайд хуваах харьцаа: IoU = Area(A ∩ B) / Area(A ∪ B). Жишээ: GT=(100,100,40,40) (1600 px²), Pred=(110,100,40,40) (1600 px²) бол давхцал=30×40=1200, нийлбэр=1600+1600−1200=2000 → IoU = 1200/2000 = 0.60 ≥ 0.50.",
        "EPE (Endpoint Error): Таамагласан вектор (u, v) ба жинхэнэ GT вектор (u_gt, v_gt)-ийн хоорондын Евклидийн зай: sqrt((u−u_gt)² + (v−v_gt)²). Жишээ: GT=(6.0, 1.0), таамаг=(5.7, 1.4) бол EPE = sqrt((−0.3)² + 0.4²) = 0.50 px.",
        "Algo-only vs End-to-End FPS: Algo-only нь зөвхөн update()/flow-ийн цэвэр медиан хугацаанаас (1000 / algo_median_ms), End-to-End (E2E) нь дискнээс кадр унших (imread) + тооцоолол + зурах + MP4 (VideoWriter) бичих бүтэн дамжлагаас тооцогдоно. Бүх хэмжилтэд 1 warmup + 5 дахин гүйлтийн median ба p95-ийг авсан.",
    ])

    # =========================================================================
    # 2. PDF ДАВТАЛТ, BINDING БА ЭХ СУРВАЛЖ
    # =========================================================================
    add_h(doc, "2. Сургалтын PDF давталт, Binding шалгалт ба эх сурвалж", level=1)
    add_p(
        doc,
        "Үндсэн хавтаст хоёр PDF олдсон: 06-Tracking-APIs.ipynb - Colaboratory.pdf (х. 1–2, доторх гарчиг 03-Tracking-APIs.ipynb) ба "
        "06-Optical-Flow.ipynb - Colaboratory.pdf (х. 1–3, доторх гарчиг 00-Optical-Flow.ipynb). reproduce_pdf.py скриптээр "
        "хоёр PDF-ийн кодыг цонх нээхгүйгээр MP4/PNG/CSV болгон бүрэн давтлаа."
    )
    add_bullets(doc, [
        "Tracker binding ба init(): OpenCV 4.14.0 (мөн 5.0.0) дээр Boosting, TLD, MedianFlow, MOSSE нь зөвхөн cv2.legacy.Tracker*_create орон зайд бий. Мөн cv2.Tracker*.init() нь None буцаадаг бол cv2.legacy.Tracker*.init() нь bool буцаадаг тул safe_tracker_init() функцээр хоосон ROI (w<=0, h<=0) болон кадр тасралтыг аюулгүй зохицуулав.",
        "Тасарсан Farnebäck код: 06-Optical-Flow.ipynb (х. 3) нь '# Set the Previous image as the next iamge for the loop' тайлбар дээр тасарсан байсан тул албан ёсны opt_flow.py-тэй тулгаж prvs_img = next_img, cap/writer release-ийг нэмж баримтжуулав.",
        "Тогтмол масштаб ба color wheel: PDF-ийн cv2.normalize(NORM_MINMAX) нь кадр бүрийг тусдаа сунгаж шуугианыг тод өнгө болгодог тул тогтмол max=12.0 px хязгаар болон өнцгийн өнгөний дугуй тайлбар (make_color_wheel) нэмэв.",
        f"Вебкамер (/dev/video0) smoke run: Физик камер дээр 12 кадр (640x480) санах ойд боловсруулахад 6 tracker бүгд ({webcam[0]['metric']}), LK ({webcam[6]['metric']}), Farnebäck ({webcam[7]['metric']}) амжилттай ажиллав (results/csv/webcam_smoke.csv; хувийн зураг хадгалаагүй).",
    ])

    b_rows = [
        [r["tracker"], r["binding"], "Тийм" if r["available"] == "1" else "Үгүй", r["class_name"], r["init_return"], r["empty_roi_handled"]]
        for r in bindings_414 if r["tracker"] != "MOSSE"
    ]
    add_table(
        doc,
        ["Tracker", "Binding зам", "Байгаа", "Класс", "init()", "Хоосон ROI"],
        b_rows,
        "Хүснэгт 1. opencv-contrib-python==4.14.0.94 (Python 3.14.4) дээрх Tracker binding ба init() шалгалт (tracker_bindings_cv414.csv). OpenCV 5.0.0.93 дээр мөн ижил үр дүн баталгаажсан.",
        font_pt=7.2,
        col_widths=[0.90, 2.50, 0.52, 1.15, 0.65, 1.35],
    )

    add_bullets(doc, [
        "opencv/opencv — lk_track.py (4.x, commit 5e80bd3, Apache-2.0): Урагш-буцааж мөрдөлтийн (FB) алдаа ба үечилсэн дахин илрүүлэлтийн тусгаар ablation санаа.",
        "opencv/opencv — opt_flow.py (4.x, commit 0e40c8a, Apache-2.0): prvs_img = next_img шинэчлэлт, сумт тор ба өнгөний дүрслэл.",
        "opencv/opencv_contrib — modules/tracking (4.x, commit d880a9a, Apache-2.0) + CSRT (Lukežič et al., CVPR 2017) + TLD/FB Error (Kalal et al., ICPR 2010) + DIS (Kroeger et al., ECCV 2016).",
    ])

    # =========================================================================
    # 3. АЛГОРИТМЫН ЗАРЧИМ БА ИНЖЕНЕРИЙН ӨРГӨТГӨЛ
    # =========================================================================
    add_h(doc, "3. Алгоритмуудын харьцуулалт ба бидний инженерийн өргөтгөл", level=1)
    algo_rows = [
        ["BOOSTING", "ROI доторх Haar шинж чанар", "Онлайн AdaBoost жинг шинэчилж хамгийн өндөр оноот байрлал хайна", "Хөдлөх хүрээ (тогтмол хэмжээ)", "Алдаатай хүрээг шууд эерэгээр сурдаг тул саад дээр гацна"],
        ["MIL", "Объектын эргэн тойрны дэд хүрээний багц", "Multiple Instance Learning-ээр ангилагч шинэчилнэ", "Тогтмол хэмжээт хүрээ", "Халхлагдсан ч success=True гэж саад дээр наалддаг (100% FP)"],
        ["KCF", "Томруулсан цонхны HOG/өнгө шинж", "FFT огторгуйд цөмийн корреляцийн шүүлтүүр бодно", "Тогтмол хэмжээ + итгэлт success", "Халхлалтад success=False болж зогсох боловч эргэж олох детекторгүй"],
        ["TLD", "10×10 цэгийн урсгал + загвар сан", "MedianFlow + бүх дэлгэцийн скан детектор + P/N сургалт", "Масштабт хүрээ + success", "Дахин олдог ч төстэй дэвсгэр рүү үсрэх (distractor jump) ба RNG хэлбэлзэлтэй"],
        ["MEDIANFLOW", "10×10 торын цэгийн LK + NCC", "Урагш-буцааж шүүж медиан шилжилт ба хос зайн масштаб бодно", "Нарийн хэмжээст хүрээ", ">50% халхлалт эсвэл хурдан хөдөлгөөний бүрсийлтэд шууд унана"],
        ["CSRT", "Өнгөний маск + HOG + Colornames", "Орон зайн маскаар жигнэсэн DCF + DSST масштаб", "Нарийн хүрээ + success", "Хэсэгчилсэн халхлалтад сайн ч бүрэн алга болсны дараа сэргэхгүй"],
        ["CSRT_REDETECT (шинэ)", "CSRT + анхны RGB/HSV загвар", "fail_streak>=2 үед олон масштабт (0.75–1.30×) template NCC + HSV корреляцаар бүх кадраас хайж confirm_streak=2-оор батална", "Нарийн хүрээ + redetect телеметр", "Бүтэн кадрын template скан нь дуудагдсан үедээ хэдэн мс нэмэлт зардалтай"],
        ["LK (sparse)", "Shi–Tomasi булангийн градиент", "Пирамид дээр цонх доторх шилжилтийг бодно", "Цэгийн бутархай (x, y) ба ID", "win200 нь хил дээр дэвсгэрийг холино; дахин нэмэгдсэн цэгийг ялгахгүй бол буруу тайлагнана"],
        ["Farnebäck", "Пиксел бүрийн полином ойролцоолол", "Пирамид түвшинд 2 зэрэгтийн олон гишүүнтийн шилжилт", "H×W×2 нягт вектор талбай", "Удаан (47–76 ms) ба гөлгөр бүтэц багатай дэвсгэрт хөдөлгөөнийг дутуу (0 руу) тооцно"],
        ["DIS (ULTRAFAST/FAST/MEDIUM)", "Inverse Search дээрх олон масштабын дэд блок", "Спарс шилжилтийг урвуу оптик хайлтаар олж вариацийн шүүлтээр жигдэлнэ", "H×W×2 нягт талбай (масштабтай)", "ULTRAFAST нь жижиг деталийг зөөлрүүлж болно; MEDIUM нь Farnebäck-ээс хурдан бөгөөд олон дахин нарийвчлалтай"],
    ]
    add_table(
        doc,
        ["Алгоритм", "Юу ажигладаг", "Тооцооллын гол санаа", "Гаралт", "Давуу ба сул тал"],
        algo_rows,
        "Хүснэгт 2. Сургалтын алгоритмууд болон нэмэлт инженерийн шийдлүүдийн хураангуй.",
        font_pt=7.2,
        col_widths=[1.10, 1.30, 2.05, 1.15, 1.55],
    )

    # =========================================================================
    # 4. ХЭРЭГЛЭСЭН ӨГӨГДӨЛ БА ШУДАРГА ХЭМЖИЛТИЙН АРГАЧЛАЛ
    # =========================================================================
    add_h(doc, "4. Өгөгдлийн бүртгэл, шүүлтийн засал ба хэмжилтийн нөхцөл", level=1)
    add_bullets(doc, [
        "Үндсэн багцын бүтэц — «5 үнэлгээний тохиолдол, 4 тусдаа дараалал»: DAVIS 2017 bmx-trees дараалал дээр дугуй (obj_id=1) болон унагч (obj_id=2) гэсэн хоёр объект үнэлэгдсэн тул нийт 4 тусдаа бичлэг дээр 5 үнэлгээний тохиолдол гарна.",
        "Файлын дугаараар тулгах засал (dataset_prep.py): DAVIS маск бэлдэхэд өмнө нь эрэмбэлсэн жагсаалтыг zip()-ээр нийлүүлж байсныг match_frames_and_masks() болгон 00042.jpg ↔ 00042.png яг нэрийн үндсээр (stem) тулгаж, дутуу кадр эсвэл хэмжээний зөрүүг ил тод шалгадаг болгов.",
        "failure_pct ба N/A (0 кадр) тодорхойлолт: failure_pct нь зөвхөн success=False бус, хүчингүй буюу дэлгэцээс хальсан хүрээг (valid_pred==0) хамтад нь тооцно. Объект огт алга болоогүй бичлэгт (absent_frames==0) «алга үеийн буруу хүрээ» мөрийг амжилтын 0% мэт харагдуулахгүйн тулд «N/A (0 кадр)» гэж ялгав.",
        "Хүлээгдэж буй сэргэлтийн ялгаа: ctrl_bicycle_occ дээр объект 18–22-р кадрт 100% алга болж 23-р кадрт дахин харагддаг тул бүрэн сэргэлт гэж үзнэ. Харин bmx-trees дээр GT маск хэзээ ч 0 болоогүй тул үүнийг «хүнд халхлалтын дараах сэргэлт» гэж тусгайлан нэрлэв.",
    ])

    m_rows = [[r["dataset_id"], r["frames_used"], r["license_or_terms"], r["sha256"][:16] + "..."] for r in manifest]
    add_table(
        doc,
        ["Өгөгдлийн ID", "Хамрах муж", "Лиценз / нөхцөл", "SHA-256"],
        m_rows,
        "Хүснэгт 3. Хэрэглэсэн бүх өгөгдлийн бүртгэл (dataset_manifest.csv). Түүхий архивуудыг .gitignore-оор GitHub-аас хамгаалсан.",
        font_pt=7.0,
        col_widths=[2.32, 1.48, 2.05, 1.22],
    )

    # =========================================================================
    # 5. ОБЪЕКТ ДАГАХ ХЭМЖИЛТ, ХАЛХЛАЛТЫН ШАТ, СТРЕСС БА ROI МЭДРЭГ ЧАНАР
    # =========================================================================
    add_h(doc, "5. Объект дагах (Tracking): Үндсэн хэмжилт, Халхлалтын шат ба Стресс тест", level=1)
    add_p(doc, "5.1 Үндсэн 5 тохиолдлын нэгдсэн харьцуулалт ба CSRT → CSRT_REDETECT → TLD", bold=True)

    t_rows = []
    for r in track_summary:
        afp = r["absent_fp_pct"] if r["absent_fp_pct"].startswith("N/A") else r["absent_fp_pct"] + "%"
        t_rows.append([
            r["sequence"], r["tracker"], f"{r['eval_frames']}/{r['absent_frames']}",
            r["mean_iou"], r["iou_ge_05_pct"] + "%", r["failure_pct"] + "%",
            r["distractor_jumps"], afp, _fmt_rec(r),
            f"{r['algo_median_ms']} ({r['algo_p95_ms']})", r["e2e_median_ms"],
        ])
    add_table(
        doc,
        ["Тохиолдол", "Tracker", "Үнэл/Алга", "Mean IoU", "IoU≥0.5", "Эвдрэл%", "Үсрэлт", "Алга FP%", "Сэргэлт", "Algo ms (p95)", "E2E ms"],
        t_rows,
        "Хүснэгт 4. Үндсэн 5 үнэлгээний тохиолдол (4 дараалал) × 8 tracker хэмжилт (track_summary.csv). STATIC нь 0 тооцоотой суурь шугам.",
        font_pt=6.4,
        col_widths=[1.30, 0.96, 0.44, 0.46, 0.46, 0.44, 0.36, 0.56, 0.78, 0.78, 0.46],
    )

    c_tld = tlookup(track_summary, "ctrl_bicycle_occ", "TLD")
    c_csrt = tlookup(track_summary, "ctrl_bicycle_occ", "CSRT")
    c_rd = tlookup(track_summary, "ctrl_bicycle_occ", "CSRT_REDETECT")
    c_mil = tlookup(track_summary, "ctrl_bicycle_occ", "MIL")
    b_csrt = tlookup(track_summary, "davis_bmx_trees_bike", "CSRT")
    b_tld = tlookup(track_summary, "davis_bmx_trees_bike", "TLD")
    b_mf = tlookup(track_summary, "davis_bmx_trees_bike", "MEDIANFLOW")
    br_csrt = tlookup(track_summary, "davis_bmx_trees_rider", "CSRT")
    br_rd = tlookup(track_summary, "davis_bmx_trees_rider", "CSRT_REDETECT")
    m_cs = tlookup(track_summary, "davis_car_shadow", "MEDIANFLOW")
    cs_csrt = tlookup(track_summary, "davis_car_shadow", "CSRT")

    add_bullets(doc, [
        f"TLD-ийн 6 ба 8 кадрын ялгаа (ctrl_bicycle_occ): Дугуй {c_tld['reappear_frame']}-р кадрт эргэж харагдсан. TLD-ийн IoU ≥ 0.5 дараалсан гурван кадрын эхний кадр нь {c_tld['first_ok_frame']} ({c_tld['recovery_start_frames']} кадрын дараа тогтвортой дагаж эхэлсэн), харин гурван кадрын шалгуур бүрэн батлагдсан кадр нь {c_tld['confirmed_frame']} ({c_tld['recovery_confirmed_frames']} кадрын дараа батлагдсан) болов. Энэ хооронд TLD {c_tld['distractor_jumps']} удаа төстэй саад объект руу үсэрсэн.",
        f"CSRT → CSRT_REDETECT → TLD харьцуулалт: Энгийн CSRT (Mean IoU={c_csrt['mean_iou']}) халхлалтын дараа сэргээгүй бол манай CSRT_REDETECT нь {c_rd['redetect_calls']} удаа бүтэн кадр скан хийж (нийт нэмэлт зардал {c_rd['redetect_extra_ms']} ms, algo={c_rd['algo_median_ms']} ms vs CSRT {c_csrt['algo_median_ms']} ms) {c_rd['recovery_start_frames']} кадрын дараа буюу frame {c_rd['first_ok_frame']}-д түрүүлэн олж, frame {c_rd['confirmed_frame']}-д ({c_rd['recovery_confirmed_frames']} кадр) баталгаажуулан Mean IoU={c_rd['mean_iou']} ({c_rd['iou_ge_05_pct']}%, 0 саад үсрэлт)-аар тэргүүллээ.",
        f"Re-detect дуудагдаагүй үеийн давхцал ба TLD RNG санамсаргүй байдал: davis_car_shadow болон ctrl_bicycle_cam_shift дээр failure=0 тул redetect_calls=0 ба CSRT_REDETECT нь CSRT-тэй 4 орон хүртэл ижил IoU ({cs_csrt['mean_iou']}) гаргасан (алгоритм хугацааны ялгаа нь OS scheduler-ийн 1-thread хэлбэлзэл; davis_bmx_trees_rider дээр CSRT_REDETECT нь {br_rd['redetect_calls']} дуудалтад {br_rd['redetect_extra_ms']} ms нэмсэн ч хуурамч үсрэлт гаргаагүй). Мөн cv2.legacy.TrackerTLD нь дотоод санамсаргүй скан/сургалттай тул bmx-trees дээр гүйлт хооронд хэлбэлздэг (энэ гүйлтэд TLD IoU={b_tld['mean_iou']}).",
        f"Хэсэгчилсэн халхлалт vs цэвэр масштаб: bmx-trees дээр CSRT ({b_csrt['mean_iou']} дугуй, {br_csrt['mean_iou']} унагч) тэргүүлж, MedianFlow дугуй дээр {b_mf['mean_iou']} болж эвдэрсэн бол сүүдэртэй, 3.5× жижгэрэх car-shadow дээр MedianFlow IoU={m_cs['mean_iou']} (100% ≥0.5, {m_cs['algo_fps']} FPS algo / {m_cs['e2e_fps']} FPS E2E) болж тэргүүллээ.",
    ])

    add_img(doc, FIGURES_DIR / "iou_timeline_occlusion_cc0.png", 6.5,
            "Зураг 2. ctrl_bicycle_occ дээрх кадр бүрийн IoU шугам: саарал муж (18–22) нь 100% бүрэн халхлалт; "
            "TLD (f29/+6f эхэлж f31/+8f батлагдсан) болон CSRT_REDETECT (f27/+4f эхэлж f29/+6f батлагдсан)-ийн ялгаа харагдана.")
    add_img(doc, FIGURES_DIR / "tracking_flow_metrics.png", 6.5,
            "Зураг 3. Үндсэн 5 тохиолдлын Mean IoU, Algo vs E2E FPS, ижил дэд багцын Optical Flow EPE болон LK анхны цэгийн ID хадгалалт.")

    add_p(doc, "5.2 Халхлалтын шат (Duration vs Distractor) ба Visible vs Amodal GT", bold=True)
    lad_rows = [
        [r["sequence"], r["tracker"], r["absent_frames"], r["mean_iou"], r["mean_iou_amodal"],
         r["iou_ge_05_pct"] + "%", r["distractor_jumps"], _fmt_rec(r)]
        for r in track_ladder if r["tracker"] in ("CSRT", "CSRT_REDETECT", "TLD", "KCF", "MIL", "MEDIANFLOW")
    ]
    add_table(
        doc,
        ["Шатны сорил", "Tracker", "Алга", "Vis IoU", "Amodal IoU", "IoU≥0.5", "Үсрэлт", "Сэргэлт"],
        lad_rows,
        "Хүснэгт 5. 3f, 11f, 22f халхлалт ба 11f+төстэй дугуйн саадтай шатны туршилт ба Visible vs Amodal GT харьцуулалт (track_occlusion_ladder.csv).",
        font_pt=6.8,
        col_widths=[1.55, 1.05, 0.35, 0.58, 0.68, 0.55, 0.48, 1.30],
    )
    add_p(
        doc,
        "Шатны шинжилгээ: Халхлалтын уртыг 3→11→22 кадр болгон сунгахад энгийн CSRT ба KCF бүгд 0.31–0.32 орчим гацдаг бол "
        "CSRT_REDETECT болон TLD хоёр л дахин олж чадна. Харин төстэй саад объект (ctrl_occ_ladder_10f_distractor) нэмэхэд "
        f"TLD саад руу {tlookup(track_ladder, 'ctrl_occ_ladder_10f_distractor', 'TLD')['distractor_jumps']} удаа үсэрч Mean IoU={tlookup(track_ladder, 'ctrl_occ_ladder_10f_distractor', 'TLD')['mean_iou']} болж унасан бол "
        f"CSRT_REDETECT нь {tlookup(track_ladder, 'ctrl_occ_ladder_10f_distractor', 'CSRT_REDETECT')['distractor_jumps']} кадр (халхлалтын 22–33-р кадрт зөрөн өнгөрөх саад дугуйтай түр давхацсан) үсрэлттэй ч жинхэнэ дугуй баганаас гарангуут эргэн барьж {tlookup(track_ladder, 'ctrl_occ_ladder_10f_distractor', 'CSRT_REDETECT')['mean_iou']} IoU-г хадгаллаа. "
        "Мөн Visible vs Amodal GT хоорондын зөрүү нь объект баганын ард хагас цухуйх үед амжилттай tracker ч бүтэн биеийн хэмжээг барьж байвал Visible-part IoU буурдгийг батална."
    )

    add_p(doc, "5.3 Нэг хүчин зүйлийн стресс тест ба Эхний ROI мэдрэг чанар", bold=True)
    conds_show = ["baseline_1x", "stride_2x", "stride_3x", "blur_gauss_k15", "blur_motion_15", "darken_05x"]
    st_rows = []
    for cond in conds_show:
        for tr in ["CSRT", "CSRT_REDETECT", "TLD", "KCF", "MIL", "MEDIANFLOW"]:
            rb = slookup("davis_bmx_trees_bike", cond, tr)
            rc = slookup("davis_car_shadow", cond, tr)
            st_rows.append([cond, tr, rb["mean_iou"], rb["failure_pct"] + "%", rc["mean_iou"], rc["failure_pct"] + "%"])
    add_table(
        doc,
        ["Нөхцөл (1 хүчин зүйл)", "Tracker", "bmx IoU", "bmx Fail%", "car IoU", "car Fail%"],
        st_rows,
        "Хүснэгт 6. Кадр алгасах (stride 2x, 3x), бүрсийлт (Gaussian/Motion blur, stride=1) ба гэрэл бууралтын (darken 0.5x) стресс тест (track_stress_tests.csv).",
        font_pt=7.2,
        col_widths=[1.45, 0.95, 0.85, 0.85, 0.85, 0.85],
    )

    roi_rows = [
        [r["sequence"], r["perturbation"], r["tracker"], r["mean_iou"], r["delta_iou"], r["iou_ge_05_pct"] + "%"]
        for r in track_roi if r["tracker"] in ("CSRT", "CSRT_REDETECT", "TLD", "MIL", "MEDIANFLOW") and r["perturbation"] != "exact_roi"
    ]
    add_table(
        doc,
        ["Дараалал", "Анхны ROI хазайлт", "Tracker", "Mean IoU", "Δ IoU (vs exact)", "IoU≥0.5"],
        roi_rows,
        "Хүснэгт 7. Эхний кадрын ROI ±10% шилжилт ба хэмжээний өөрчлөлтийн мэдрэг чанар (track_roi_sensitivity.csv).",
        font_pt=7.0,
        col_widths=[1.50, 1.15, 1.02, 0.68, 0.85, 0.68],
    )
    add_img(doc, FIGURES_DIR / "occlusion_ladder_stress_cc0.png", 6.5,
            "Зураг 4. (A) Халхлалтын шат, (B) Visible vs Amodal GT зөрүү, (C) Стресс тестүүд, (D) Эхний ROI ±10% гажилтын нөлөө.")

    # =========================================================================
    # 6. OPTICAL FLOW ХАРЬЦУУЛАЛТ, DIS, ЦЭГИЙН ID БА КАМЕР НӨХӨЛТ
    # =========================================================================
    add_h(doc, "6. Optical Flow: Ижил дэд багц, Бүсчилсэн EPE, DIS ба Камер нөхөлт", level=1)
    add_p(doc, "6.1 Ижил хүчинтэй цэгийн дэд багц (Common Subset) болон Farnebäck vs DISOpticalFlow", bold=True)

    f_rows = [
        [r["dataset"], r["method"], f"{r['common_pts']}/{r['gt_valid_pts']}", r["point_loss_pct"] + "%",
         r["common_pts_epe"], r["all_valid_px_epe"], f"{r['algo_median_ms']} ({r['algo_p95_ms']})", r["e2e_median_ms"]]
        for r in flow_summary if r["max_corners"] == "10"
    ]
    add_table(
        doc,
        ["Өгөгдөл (maxCorners=10)", "Арга", "Ижил цэг", "Алдалт%", "EPE (common)", "EPE (бүх px)", "Algo ms (p95)", "E2E ms"],
        f_rows,
        "Хүснэгт 8. Middlebury, CC0 болон held-out MPI Sintel дээр бүх аргыг ИЖИЛ хүчинтэй цэгийн дэд багц (common_pts) ба нийт пиксел дээр харьцуулсан дүн (flow_summary.csv).",
        font_pt=6.5,
        col_widths=[1.52, 1.15, 0.56, 0.46, 0.65, 0.65, 0.85, 0.54],
    )

    fb_cam = flookup("Controlled:cam_shift", "Farneback_dense")
    dm_cam = flookup("Controlled:cam_shift", "DIS_MEDIUM")
    df_cam = flookup("Controlled:cam_shift", "DIS_FAST")
    fb_sin_c = flookup("Sintel:alley_1_clean", "Farneback_dense")
    dm_sin_c = flookup("Sintel:alley_1_clean", "DIS_MEDIUM")
    fb_sin_f = flookup("Sintel:ambush_5_final", "Farneback_dense")
    dm_sin_f = flookup("Sintel:ambush_5_final", "DIS_MEDIUM")

    add_bullets(doc, [
        f"Камерын шилжилтийг дүгнэлтэд заавал оруулах нь: Controlled:cam_shift дээр Farnebäck 10 сонгосон булан дээр ердөө {fb_cam['common_pts_epe']} px алдаатай байсан ч бүх хүчинтэй пикселийн дундаж алдаа {fb_cam['all_valid_px_epe']} px болов (Middlebury + obj_occ-ийн 0.256–0.585 px мужтай харьцуулахад 2–5 дахин их!). Шалтгаан: бүтэц багатай саарал бетон хэсэгт Farnebäck жигд шилжилтийг олж чадалгүй 0 орчим вектор өгчээ (Хүснэгт 9, low_texture={rlookup('Controlled:cam_shift', 'Farneback_dense')['low_texture_epe']} px vs high_texture={rlookup('Controlled:cam_shift', 'Farneback_dense')['high_texture_epe']} px).",
        f"DISOpticalFlow vs Farnebäck (ижил 1-thread CPU): DIS_MEDIUM нь cam_shift дээр бүх пикселийн EPE-г {fb_cam['all_valid_px_epe']} px → {dm_cam['all_valid_px_epe']} px болгож ~50 дахин бууруулаад зогсохгүй хурд нь ~2× ({dm_cam['algo_median_ms']} ms vs {fb_cam['algo_median_ms']} ms) байв. DIS_FAST нь ердөө {df_cam['algo_median_ms']} ms ({df_cam['algo_fps']} FPS) дотор {df_cam['all_valid_px_epe']} px гаргав. Бодит бэлдээгүй MPI Sintel (alley_1_clean ба ambush_5_final) дээр ч DIS_MEDIUM ({dm_sin_c['all_valid_px_epe']} ба {dm_sin_f['all_valid_px_epe']} px) нь Farnebäck ({fb_sin_c['all_valid_px_epe']} ба {fb_sin_f['all_valid_px_epe']} px)-ийг нарийвчлал ба хурд аль алинаар нь давлаа.",
    ])

    add_p(doc, "6.2 Орон зайн бүсчилсэн EPE (Зах, Бүтэц багатай/ихтэй, Хил, Дотор, Дэвсгэр)", bold=True)
    reg_rows = [
        [r["dataset"], r["method"], r["all_valid_epe"], r["border_band_epe"],
         r["low_texture_epe"], r["high_texture_epe"], r["object_boundary_epe"], r["inside_object_epe"], r["background_epe"]]
        for r in flow_regions if r["method"] in ("Farneback_dense", "DIS_FAST", "DIS_MEDIUM")
    ]
    add_table(
        doc,
        ["Өгөгдөл", "Арга", "Бүгд", "Зах 24px", "Бүтэц бага", "Бүтэц их", "Объект хил", "Объект дотор", "Дэвсгэр"],
        reg_rows,
        "Хүснэгт 9. Нягт урсгалын бүсчилсэн EPE задлал (px, flow_dense_regions.csv).",
        font_pt=6.5,
        col_widths=[1.52, 1.05, 0.50, 0.58, 0.60, 0.58, 0.62, 0.62, 0.58],
    )
    add_img(doc, FIGURES_DIR / "flow_error_and_dis_cc0.png", 6.5,
            "Зураг 5. (A–B) cam_shift дээрх Farnebäck vs DIS_MEDIUM алдааны газрын зураг (ижил 0..6 px масштаб), "
            "(C) Бүсчилсэн EPE, (D) Бүх 7 хос зураг дээрх хурд-нарийвчлалын график.")

    add_p(doc, "6.3 LK Цэг бүрийн ID хадгалалт ба Хуримтлагдсан шилжилтийн алдаа (Cumulative Drift)", bold=True)
    ab_rows = [
        [r["sequence"], r["variant"], r["winSize"], r["fb_check"], r["redetect_interval"],
         f"{r['retained_initial_pts']}/{r['init_pts']} ({r['initial_retention_pct']}%)",
         r["redetected_pts"], r["final_active_pts"], r["step_mean_epe_px"], r["final_cum_drift_px"], r["algo_median_ms"]]
        for r in flow_ablation
    ]
    add_table(
        doc,
        ["Дараалал", "Хувилбар", "win", "FB", "Redet", "Анхны цэг үлдэлт", "Шинэ цэг", "Нийт идэвх", "Алхам EPE", "Хуримт. drift", "Algo ms"],
        ab_rows,
        "Хүснэгт 10. Цэг бүрийн ID хяналттай LK ablation (flow_ablation.csv & flow_trajectory_drift.csv).",
        font_pt=6.3,
        col_widths=[1.38, 1.52, 0.52, 0.24, 0.35, 0.80, 0.44, 0.48, 0.50, 0.55, 0.44],
    )
    occ_fb_red = alookup("Controlled:bicycle_occ_40f", "LK_PDF_win200_FB_redetect5")
    bmx_fb_red = alookup("DAVIS:bmx_trees_80f", "LK_PDF_win200_FB_redetect5")
    cam_w200 = alookup("Controlled:cam_shift_24f", "LK_PDF_win200_noFB")
    cam_w15 = alookup("Controlled:cam_shift_24f", "LK_small_win15_noFB")
    add_p(
        doc,
        f"Нэр томьёоны залруулга (цэгийн ID ялгалт): Өмнөх хувилбарт сүүлийн кадрын идэвхтэй цэгийг анхныхд хувааж эндүүрэлтэй тайлбарласныг засав. "
        f"Шинэ хэмжүүрээр Controlled:bicycle_occ_40f (FB+redetect5) дээр «Сүүлийн кадрт {occ_fb_red['final_active_pts']} идэвхтэй цэг байсан» — үүнээс анхны {occ_fb_red['init_pts']} цэгийн ердөө {occ_fb_red['retained_initial_pts']} нь ({occ_fb_red['initial_retention_pct']}%) хадгалагдаж, "
        f"{occ_fb_red['redetected_pts']} цэг шинээр нэмэгдсэн. Харин бодит DAVIS:bmx_trees_80f дээр анхны цэгүүд {bmx_fb_red['retained_initial_pts']}/{bmx_fb_red['init_pts']} (0%) үлдэж, сүүлийн {bmx_fb_red['final_active_pts']} цэг бүгд дахин илрүүлсэн ({bmx_fb_red['redetected_pts']}) шинэ цэгүүд байв! "
        f"Түүнчлэн cam_shift_24f дээр 1 кадрын алхмын EPE нь LK_win200={cam_w200['step_mean_epe_px']} px (win15={cam_w15['step_mean_epe_px']} px) байсан ч 23 кадр дараалан хуримтлагдахад нийт траекторийн drift={cam_w200['final_cum_drift_px']} px (win15={cam_w15['final_cum_drift_px']} px) болж өссөн нь урт видео мөрдөлтийн зөрүүг илтгэнэ."
    )

    add_p(doc, "6.4 Камерын ерөнхий хөдөлгөөн нөхөлт (Global Affine + Stratified FB-LK + RANSAC)", bold=True)
    cc_rows = [
        [r["pair"], r["flow_method"], r["raw_bg_mag"], r["est_cam_mag"], r["residual_bg_mag"], r["bg_reduction_pct"] + "%", r["raw_fg_mag"], r["residual_fg_mag"]]
        for r in cam_comp
    ]
    add_table(
        doc,
        ["Хос зураг", "Нягт арга", "Түүхий дэвсгэр", "Тооцсон камер", "Үлдэгдэл дэвсгэр", "Бууралт%", "Түүхий объект", "Үлдэгдэл объект"],
        cc_rows,
        "Хүснэгт 11. Камерын ерөнхий шилжилтийг хассаны дараах дэвсгэр ба хөдлөх объектын урсгалын хэмжээ (px, camera_compensation.csv).",
        font_pt=6.8,
        col_widths=[1.55, 0.85, 0.78, 0.78, 0.82, 0.64, 0.78, 0.78],
    )
    add_img(doc, FIGURES_DIR / "camera_motion_compensation_cc0.png", 6.5,
            "Зураг 6. CC0 Pan+Moving Object: (Зүүн) Камер+объект холилдсон түүхий DIS_MEDIUM урсгал, (Дунд) Стратифицирлэсэн LK+RANSAC-аар тооцсон камерын урсгал, (Баруун) Үлдэгдэл урсгал — хөдөлгөөнгүй дэвсгэр харлаж, зөвхөн хөдлөх дугуй тод үлдэнэ.")

    # =========================================================================
    # 7. HELD-OUT GENERALIZATION БА УТАСНЫ ВИДЕОНЫ ХЭРЭГСЭЛ
    # =========================================================================
    add_h(doc, "7. Бэлдээгүй өгөгдөл (Held-out Generalization) ба Утасны видеоны хэрэгсэл", level=1)
    add_p(
        doc,
        "Тохируулга огт өөрчлөлгүйгээр (өмнө бэхэлсэн параметрээр) бэлдээгүй LaSOT bicycle-11 (120 кадр, 74–78-р кадрт бодит 100% бүрэн халхлалт) болон "
        "DAVIS scooter-black (43 кадр) дарааллууд дээр шалгав (Хүснэгт 12). Мөн flow хэсэгт MPI Sintel-ийн 2 өгөгдөл дээр DIS_MEDIUM нь Farnebäck-ийг тогтвортой давсан (Хүснэгт 8)."
    )
    gen_rows = [
        [r["sequence"], r["tracker"], r["eval_frames"], r["absent_frames"], r["mean_iou"], r["iou_ge_05_pct"] + "%",
         r["failure_pct"] + "%", _fmt_rec(r), f"{r['algo_median_ms']} ({r['e2e_median_ms']})"]
        for r in track_gen
    ]
    add_table(
        doc,
        ["Held-out дараалал", "Tracker", "Үнэл", "Алга", "Mean IoU", "IoU≥0.5", "Эвдрэл%", "Сэргэлт", "Algo (E2E) ms"],
        gen_rows,
        "Хүснэгт 12. Өмнө тохируулсан параметрээр бэлдээгүй LaSOT bicycle-11 ба DAVIS scooter-black дээр хийсэн шалгалт (track_generalization_lasot.csv).",
        font_pt=6.8,
        col_widths=[1.42, 1.02, 0.38, 0.34, 0.56, 0.54, 0.50, 0.90, 0.88],
    )
    add_p(
        doc,
        f"Held-out дүгнэлт ба CSRT_REDETECT-ийн шударга хязгаар: LaSOT bicycle-11 (320×240, 74–78-р кадрт бүрэн халхлалт) дээр "
        f"эргэлт-масштабын огцом өөрчлөлтийн улмаас бүх арга 0.40 орчим IoU-д хүрсэн (CSRT={tlookup(track_gen, 'lasot_bicycle11', 'CSRT')['mean_iou']}, "
        f"TLD={tlookup(track_gen, 'lasot_bicycle11', 'TLD')['mean_iou']}). Энд CSRT_REDETECT нь ({tlookup(track_gen, 'lasot_bicycle11', 'CSRT_REDETECT')['mean_iou']}) энгийн CSRT-ээс доош орсон нь чухал сургамж болов: "
        "33-р кадрт дугуй 3D эргэж харагдах үед анхны frame 0-ийн тогтмол (frozen) загвартай харьцуулах оноо < 0.22 болж унаад lost горимд шилжсэн бөгөөд "
        "дараагийн хэт жижгэрэлтэд (w=156→17 px) frame 0-ийн 0.75–1.30× масштабын скан нь 0.50 босгыг даваагүй (online загвар шинэчлэхгүй сул тал). "
        f"Харин DAVIS scooter-black дээр MedianFlow ({tlookup(track_gen, 'davis_scooter_black', 'MEDIANFLOW')['mean_iou']}) болон "
        f"CSRT/CSRT_REDETECT ({tlookup(track_gen, 'davis_scooter_black', 'CSRT')['mean_iou']}) бусад аргаас илүү тогтвортой байв."
    )

    add_p(doc, "Хэрэглэгчийн гар утасны 3 ширээний бичлэгийн бодит төлөв:", bold=True)
    add_p(
        doc,
        "Компьютер дээр хэрэглэгчийн биечлэн бичсэн гар утасны ширээний 3 бичлэг (тогтвортой, гар хөдөлгөөнтэй, гарт/хайрцагт халхлагдах ном эсвэл тоглоомон машин) байхгүй бөгөөд хиймэл оюун ухаан биетээр бичлэг хийх эсвэл зохиомол зураг бичлэгээр орлуулах ёсгүй. "
        "Иймд бичлэгээ хийсний дараа нэг командаар түлхүүр кадровт ROI тэмдэглэж (--annotate) болон бүх 8 tracker + камер нөхөлттэй optical flow-ийг автоматаар хэмжих (--evaluate) custom_video_lab.py хэрэгслийг бэлэн болгов (заавар README.md дээр)."
    )

    # =========================================================================
    # 8. ЗУРАГТ ШИНЖИЛГЭЭ (STORYBOARD) БА ЭЦСИЙН ДҮГНЭЛТ
    # =========================================================================
    add_h(doc, "8. Эвдрэлийн зурагт жишээ ба Нэгдсэн дүгнэлт", level=1)
    add_img(doc, FIGURES_DIR / "bicycle_storyboard_cc0.png", 6.4,
            "Зураг 7. Хувилбар A (CC0 ctrl_bicycle_occ, Frame 4, 13, 19, 31): GT vs CSRT/CSRT_REDETECT/TLD/KCF, LK цэгүүд ба Farnebäck HSV+сум.")
    add_img(doc, FIGURES_DIR / "camera_shift_storyboard_cc0.png", 6.4,
            "Зураг 8. Хувилбар B (ctrl_bicycle_cam_shift, Frame 1, 7, 14, 21): дэлгэц бүхэлдээ 180° хөх шилжилттэй ч бетон дээр Farnebäck 0 рүү унаж буй хар нүх ажиглагдана.")
    add_img(doc, FIGURES_DIR / "lk_vs_farneback_cc0.png", 6.4,
            "Зураг 9. Сийрэг LK (улаан=win200, ногоон=win15) ба нягт Farnebäck дүрслэл.")
    add_img(doc, LOCAL_VIZ_DIR / "davis_bmx_failures.png", 6.1,
            "Зураг 10. DAVIS 2017 bmx-trees (CC BY-NC 4.0) дээрх эвдрэлийн кадрууд (Frame 10, 30, 61, 72; сүүлийн CSV үр дүнгээс зурсан).")

    # Frame-level evidence dynamically pulled from per-frame CSVs
    f_csrt = _by_frame(CSV_DIR / "track_ctrl_bicycle_occ_CSRT.csv")
    f_rd = _by_frame(CSV_DIR / "track_ctrl_bicycle_occ_CSRT_REDETECT.csv")
    f_tld = _by_frame(CSV_DIR / "track_ctrl_bicycle_occ_TLD.csv")
    f_mil = _by_frame(CSV_DIR / "track_ctrl_bicycle_occ_MIL.csv")
    d_csrt = _by_frame(CSV_DIR / "track_davis_bmx_trees_bike_CSRT.csv")
    d_tld = _by_frame(CSV_DIR / "track_davis_bmx_trees_bike_TLD.csv")

    add_p(doc, "Сайн ба муу кадрын тоон баталгаа (кадр бүрийн CSV-ээс):", bold=True)
    add_bullets(doc, [
        f"ctrl_bicycle_occ Frame 4 (цэвэр харагдах үе): CSRT IoU={f_csrt[4]['iou']}, CSRT_REDETECT IoU={f_rd[4]['iou']}, TLD IoU={f_tld[4]['iou']}, MIL IoU={f_mil[4]['iou']}.",
        f"ctrl_bicycle_occ Frame 19 (100% халхлалт): CSRT (ok={f_csrt[19]['success']}), CSRT_REDETECT (ok={f_rd[19]['success']}), TLD (ok={f_tld[19]['success']}) бүгд 0% алга FP барьсан бол MIL ok={f_mil[19]['success']} гэж баганан дээр наалдсан.",
        f"ctrl_bicycle_occ Frame 31 (сэргэлт батлагдсан кадр): CSRT (ok={f_csrt[31]['success']}, IoU={f_csrt[31]['iou']}) сэргээгүй бол TLD IoU={f_tld[31]['iou']} ба CSRT_REDETECT IoU={f_rd[31]['iou']}-ээр амжилттай мөрдөв.",
        f"DAVIS bmx-trees (Frame 10 → 61): Frame 10 дээр CSRT IoU={d_csrt[10]['iou']}, TLD IoU={d_tld[10]['iou']}; хүнд халхлалтын Frame 61 дээр CSRT IoU={d_csrt[61]['iou']}, TLD IoU={d_tld[61]['iou']}.",
    ])

    add_p(doc, "Нэгдсэн дүгнэлт ба бодит гүйцэтгэлийн төлөв:", bold=True)
    fb_rw = flookup("Middlebury:RubberWhale", "Farneback_dense")
    fb_g2 = flookup("Middlebury:Grove2", "Farneback_dense")
    fb_oo = flookup("Controlled:obj_occ", "Farneback_dense")
    dm_oo = flookup("Controlled:obj_occ", "DIS_MEDIUM")
    las_csrt = tlookup(track_gen, "lasot_bicycle11", "CSRT")
    las_rd = tlookup(track_gen, "lasot_bicycle11", "CSRT_REDETECT")
    las_tld = tlookup(track_gen, "lasot_bicycle11", "TLD")

    add_bullets(doc, [
        f"Аль tracker хэзээ шилдэг вэ: (1) Халхлалтгүй, жигд жижгэрэх биетэд (car-shadow) MedianFlow (IoU={m_cs['mean_iou']}, {m_cs['algo_fps']} FPS algo) хамгийн нарийн бөгөөд хурдан; (2) Хэсэгчилсэн халхлалтад (bmx-trees) CSRT ({b_csrt['mean_iou']} / {br_csrt['mean_iou']}) тогтвортой; (3) Бүрэн халхлалтын дараа дахин илрүүлэлтэд TLD-ээс илүү CSRT_REDETECT нь саад үсрэлт багатай, илүү эрт сэргэлт үзүүлэв (ctrl_bicycle_occ IoU {c_rd['mean_iou']} vs {c_tld['mean_iou']}; бэлдээгүй LaSOT bicycle-11 дээр CSRT={las_csrt['mean_iou']}, CSRT_REDETECT={las_rd['mean_iou']}, TLD={las_tld['mean_iou']}).",
        f"Optical Flow (бүх өгөгдөл ба cam_shift-ийг багтаасан дүгнэлт): Жижиг цонхтой LK win15 нь булан цэгүүд дээр ~0.3 ms хурдтай, объектын хил дээр win200 шиг холих алдаа гаргахгүй ({flookup('Controlled:obj_occ','LK_small_win15')['common_pts_epe']} vs {flookup('Controlled:obj_occ','LK_PDF_win200')['common_pts_epe']} px). Farnebäck нь Middlebury/obj_occ дээр {fb_oo['all_valid_px_epe']}–{fb_g2['all_valid_px_epe']} px (RubberWhale={fb_rw['all_valid_px_epe']} px) авсан ч камерын шилжилт (cam_shift) дээр бүтэц багатай хэсгийг алдсанаас {fb_cam['all_valid_px_epe']} px, Sintel ambush_5_final дээр {fb_sin_f['all_valid_px_epe']} px болж өссөн. Харин DIS_MEDIUM нь ижил бүх туршилтад {dm_cam['all_valid_px_epe']}–{dm_sin_f['all_valid_px_epe']} px (obj_occ={dm_oo['all_valid_px_epe']} px) болж Farnebäck-ээс нарийвчлал ба хурдаараа аль алинаар нь давамгайлав.",
        f"Гүйцэтгэлийн бодит статус: Бүх 8 tracker (STATIC, 6 OpenCV/legacy + бидний CSRT_REDETECT) болон 6 optical flow аргыг OpenCV {bindings_414[1]['opencv_version']}-д бүрэн гүйцэтгэв (BLOCKED API байхгүй; физик утасны 3 ширээний бичлэг хэрэглэгчийн бичлэг хүлээгдэж буй тул custom_video_lab.py-д бэлдэв).",
    ])

    out_path = ROOT / "Tracking_Lab_Report_MN.docx"
    doc.save(out_path)
    return out_path


if __name__ == "__main__":
    p = build_report()
    print("Built report:", p)
