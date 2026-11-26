"""Build beginner-friendly Mongolian laboratory report Tracking_Lab_Report_MN.docx directly from CSVs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from config import CSV_DIR, FIGURES_DIR, LOCAL_VIZ_DIR, ROOT

BLACK = RGBColor(0x00, 0x00, 0x00)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _set_run_black(run, size: int = 10, bold: bool = False, italic: bool = False) -> None:
    run.font.name = "DejaVu Sans"
    run.font.size = Pt(size)
    run.font.color.rgb = BLACK
    run.bold = bold
    run.italic = italic


def add_h(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    sizes = {0: 16, 1: 13, 2: 11}
    for r in h.runs:
        _set_run_black(r, size=sizes.get(level, 11), bold=True)


def add_p(doc: Document, text: str, size: int = 10, bold: bool = False, italic: bool = False) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    _set_run_black(r, size=size, bold=bold, italic=italic)


def add_bullets(doc: Document, items: List[str], size: int = 10) -> None:
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
    font_pt: int = 8,
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

    add_p(doc, caption, size=9, italic=True)


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
    _set_run_black(r, size=9, italic=True)


def build_report(repo_url_note: str = "https://github.com/JinLight0818/cv-tracking-optical-flow-lab (private)") -> Path:
    track_summary = _read_csv(CSV_DIR / "track_summary.csv")
    flow_summary = _read_csv(CSV_DIR / "flow_summary.csv")
    flow_ablation = _read_csv(CSV_DIR / "flow_ablation.csv")
    manifest = _read_csv(CSV_DIR / "dataset_manifest.csv")
    bindings_414 = _read_csv(CSV_DIR / "tracker_bindings_cv414.csv")
    failures = _read_csv(CSV_DIR / "track_failures.csv")
    webcam = _read_csv(CSV_DIR / "webcam_smoke.csv")

    # Quick lookups from CSV so narrative matches tables identically
    def tlookup(seq: str, tr: str) -> Dict[str, str]:
        return next(r for r in track_summary if r["sequence"] == seq and r["tracker"] == tr)

    def flookup(ds: str, method: str, mc: str = "10") -> Dict[str, str]:
        return next(r for r in flow_summary if r["dataset"] == ds and r["method"] == method and r["max_corners"] == mc)

    def alookup(seq: str, var: str) -> Dict[str, str]:
        return next(r for r in flow_ablation if r["sequence"] == seq and r["variant"] == var)

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "DejaVu Sans"
    normal.font.size = Pt(10)
    normal.font.color.rgb = BLACK

    for sec in doc.sections:
        sec.top_margin = Inches(0.65)
        sec.bottom_margin = Inches(0.65)
        sec.left_margin = Inches(0.65)
        sec.right_margin = Inches(0.65)

    add_h(doc, "Объект дагах (Object Tracking) ба Optical Flow лабораторийн тайлан", level=0)
    add_p(doc, "Орчин: Linux (x86_64), Intel(R) Core(TM) 5 120U, Python 3.14.4, opencv-contrib-python==4.14.0.94 (1 thread)", size=9, italic=True)
    add_p(doc, f"Төслийн эх код ба баримт: {repo_url_note}", size=9, italic=True)

    # =========================================================================
    # 1. ЗОРИЛГО БА ҮНДСЭН ОЙЛГОЛТ
    # =========================================================================
    add_h(doc, "1. Зорилго ба анхан шатны ойлголт", level=1)
    add_p(
        doc,
        "Энэхүү лабораторийн зорилго нь дараалсан видео кадруудаас сонгосон объектыг дагах OpenCV tracker-үүд "
        "(Boosting, MIL, KCF, TLD, MedianFlow, CSRT) болон дүрсний шилжилтийг вектор талбайгаар хэмжих "
        "optical flow аргуудыг (Shi–Tomasi + pyramidal Lucas–Kanade, Farnebäck dense) хоёр сургалтын PDF "
        "материалын дагуу давтан ажиллуулж, бодит болон хяналттай өгөгдөл дээр ижил нөхцөлөөр хэмжиж "
        "харьцуулах явдал юм."
    )
    add_p(doc, "Анх удаа компьютер харааг үзэж байгаа уншигч дугуйн бичлэг дээр дараах гурван асуултаар ялгааг шууд ойлгож болно:", bold=True)
    add_bullets(doc, [
        "1) Дугуй одоо хаана байна? — Дугуйг бүхэлд нь багтаан хүрээлэх тэгш өнцөгт (bounding box) гаргах нь объект дагагч (object tracker)-ийн хариу юм.",
        "2) Дугуйн хүрээний хэдэн танигдах цэг хаашаа шилжив? — Хүчтэй булан/ирмэгийн цөөн цэгийг кадр дамжуулан мөрдөх нь сийрэг урсгал буюу Lucas–Kanade (LK) аргын жишээ юм.",
        "3) Дугуй болон орчны пикселүүд хаашаа, хэр их шилжив? — Дүрсний пиксел бүрд хөдөлгөөний чиглэл, хэмжээтэй вектор оноох нь нягт урсгал буюу Farnebäck аргын жишээ юм.",
    ])

    add_p(doc, "Үндсэн нэр томьёо ба хэмжүүрийн тодорхойлолт (энгийн тооцооны жишээтэй):", bold=True)
    add_bullets(doc, [
        "ROI (Region of Interest — сонирхлын муж): Эхний кадр дээр дагахыг хүссэн объектоо тэмдэглэсэн анхны тэгш өнцөгт хүрээ (x, y, w, h). Энд x, y нь зүүн дээд булан, w нь өргөн, h нь өндөр (пикселээр).",
        "Ground Truth (GT — бодит үнэн шошго): Кадр бүрд объектын жинхэнэ байрлал (эсвэл пиксел бүрийн жинхэнэ хөдөлгөөний вектор). Бидний таамаглалыг шалгах эталон.",
        "IoU (Intersection over Union — давхцлын харьцаа): Таамагласан хүрээ ба GT хүрээ хоёрын давхцсан талбайг тэдгээрийн нийлбэр (давхцлыг давхар тоолохгүй) талбайд хуваасан тоо (0-ээс 1 хооронд, 1 бол яг таарсан). Жишээ нь GT=(100, 100, 40, 40) ба Tracker=(110, 100, 40, 40) бол тус бүрийн талбай 40×40=1600, давхцсан хэсгийн өргөн 30 ба өндөр 40 тул давхцал=1200, нийлбэр=1600+1600−1200=2000, иймд IoU = 1200/2000 = 0.60 ≥ 0.50 (амжилттай дагалт).",
        "EPE (Endpoint Error — төгсгөлийн цэгийн алдаа): Таамагласан хөдөлгөөний вектор (u, v) ба жинхэнэ GT вектор (u_gt, v_gt) хоёрын үзүүр хоорондын Евклидийн зай (пикселээр). Жишээ нь GT=(6.0, 1.0), таамаглал=(5.7, 1.4) бол алдааны зөрүү Δu=−0.3, Δv=+0.4 ба EPE = sqrt((−0.3)^2 + 0.4^2) = sqrt(0.09 + 0.16) = 0.50 пиксел.",
        "FPS (Frames Per Second — секундэд боловсруулах кадрууд): Алгоритм 1 кадр боловсруулах медиан хугацаанаас тооцогдоно: FPS = 1000 / median_ms. Жишээ нь 20 ms/frame бол 50.0 FPS.",
    ])

    # =========================================================================
    # 2. PDF ДАВТАЛТЫН ҮР ДҮН БА GITHUB ЭХ СУРВАЛЖ
    # =========================================================================
    add_h(doc, "2. Сургалтын PDF давталт, binding шалгалт ба GitHub эх сурвалж", level=1)
    add_p(
        doc,
        "Эх хавтаст хоёр сургалтын PDF файл олдсон: (1) 06-Tracking-APIs.ipynb - Colaboratory.pdf (х. 1–2, доторх гарчиг 03-Tracking-APIs.ipynb), "
        "(2) 06-Optical-Flow.ipynb - Colaboratory.pdf (х. 1–3, доторх гарчиг 00-Optical-Flow.ipynb). Локал хавтаст .ipynb эх файл байхгүй тул "
        "хоёр PDF-ийн кодыг мөр бүрээр шалгаж, GUI-гүй (headless) горимоор MP4/PNG/CSV гаргадаг reproduce_pdf.py хэлбэрээр давтав."
    )
    add_p(doc, "PDF кодод хийсэн зайлшгүй засвар ба захын тохиолдлууд:", bold=True)
    add_bullets(doc, [
        "Tracker binding ба init() ялгаа: OpenCV 4.14.0 болон 5.0.0 дээр Boosting, TLD, MedianFlow (мөн MOSSE) нь үндсэн cv2 орон зайд байхгүй бөгөөд cv2.legacy.Tracker*_create-д шилжсэн. Харин MIL, KCF, CSRT нь cv2 ба cv2.legacy аль алинд нь бий. Анхаарах нарийн ялгаа: cv2.Tracker*.init() нь None буцаадаг бол cv2.legacy.Tracker*.init() нь True/False буцаана. Иймд safe_tracker_init() функц бичиж аль ч binding-ийг зөв таньж, хоосон ROI (w<=0 эсвэл h<=0) болон хоосон кадрыг эхэнд нь тасалдаг болгов.",
        "Farnebäck тасарсан код: 06-Optical-Flow.ipynb - Colaboratory.pdf (х. 3) нь '# Set the Previous image as the next iamge for the loop' гэсэн тайлбарын доор тасарсан (prvs_img = next_img, cap.release(), cv2.destroyAllWindows() мөрүүд дутуу). Албан ёсны opt_flow.py жишээтэй тулгаж давталтын төгсгөлд prvs_img = next_img, try/finally дотор cap.release() ба writer.release() нэмэв.",
        "Farnebäck дүрслэлийн тогтмол масштаб: PDF-ийн cv2.normalize(..., NORM_MINMAX) нь кадр бүрийг тусад нь сунгадаг тул хөдөлгөөнгүй кадрт шуугианыг тод өнгө болгож, өөр кадруудын хурдыг нүдээр харьцуулах боломжгүй болгодог. Бид тогтмол дээд хязгаар (max=12.0 px эсвэл 6.0 px) ба өнцгийн өнгөний дугуй тайлбар (color wheel legend) нэмэв.",
        "LK бутархай координат ба цэг дуусах: Тооцоонд float32 координатыг хадгалж, зөвхөн зурах үед int(round(...)) болгов. Мөн status==1 ба кадрын хязгаараас халих эсэхийг шалгаж, цэг 0 болсон ч програм эвдрэлгүй ажилладаг болгов.",
        "Вебкамер (/dev/video0) smoke run: Физик камер холбоотой тул reproduce_pdf.py --mode smoke-webcam ажиллуулж 12 кадр (640x480) санах ойд боловсруулахад 6 tracker бүгд 12/12 success, LK 10/10 цэг, Farnebäck mean_mag=0.0515 px-тэй амжилттай ажиллав (хувийн зураг хадгалаагүй, results/csv/webcam_smoke.csv-д бүртгэсэн).",
    ])

    b_rows = [
        [r["tracker"], r["binding"], "Тийм" if r["available"] == "1" else "Үгүй", r["class_name"], r["init_return"], r["empty_roi_handled"]]
        for r in bindings_414 if r["tracker"] != "MOSSE"
    ]
    add_table(
        doc,
        ["Tracker", "Binding зам", "Байгаа эсэх", "Класс", "init() буцаалт", "Хоосон ROI шалгалт"],
        b_rows,
        "Хүснэгт 1. Clean орчин (opencv-contrib-python==4.14.0.94, Python 3.14.4) дээрх Tracker binding шалгалт (results/csv/tracker_bindings_cv414.csv). OpenCV 5.0.0.93 орчинд ч яг ижил 6 legacy + 3 үндсэн binding баталгаажсан.",
        font_pt=8,
        col_widths=[0.95, 2.25, 0.70, 1.25, 0.75, 1.30],
    )

    add_p(doc, "GitHub эх код ба судалгааны өгүүллээс авсан гол санаанууд:", bold=True)
    add_bullets(doc, [
        "opencv/opencv — lk_track.py (branch 4.x, commit 5e80bd3, Apache-2.0): Урагш-буцааж мөрдөх шалгалт (||p0 - p0r|| < 1.0) нь халхлагдаж гацсан цэгийг шүүдэг бол үе үе шилжүүлэлтээр шинэ цэг нэмэх (detect_interval=5) нь цэг дуусахаас сэргийлдгийг тус тусдаа ablation туршилтаар шалгах санааг авлаа.",
        "opencv/opencv — opt_flow.py (branch 4.x, commit 0e40c8a, Apache-2.0): Тасарсан Farnebäck давталтыг prvs_img = next_img-ээр гүйцээх, сумт тор (draw_flow) болон тогтмол масштабаар HSV хурдыг дүрслэх аргыг авч ашиглав.",
        "opencv/opencv_contrib — modules/tracking (branch 4.x, commit d880a9a, Apache-2.0): C++ эх код (trackerMedianFlow.cpp, tldTracker.cpp, trackerCSRT.cpp, tracker_mil.cpp) шалгаснаар MedianFlow нь 10×10 тороор FB+NCC шүүлт ба хос зайн масштабыг боддог, TLD нь MedianFlow+детектор+P/N сургалтыг хослуулдгийг тогтоов.",
        "Lukežič et al., CVPR 2017 (CSRT / CSR-DCF): Орон зайн найдвартай байдлын маск ба сувгийн жин (HOG + Colornames) нь тэгш өнцөгт бус объект болон хэсэгчилсэн халхлалтад KCF-ээс илүү тогтвортой болохыг DAVIS bmx-trees туршилтад батлав.",
        "Kalal et al., ICPR 2010 (Forward–Backward Error): Урагш ба буцах мөрдөлтийн зөрүү нь хяналтгүй орчинд эвдрэлийг автоматаар илрүүлэх найдвартай дохио болохыг LK цэгийн болон MedianFlow-ийн шинжилгээнд ашиглав.",
    ])

    # =========================================================================
    # 3. АЛГОРИТМЫН ХАРЬЦУУЛАЛТ
    # =========================================================================
    add_h(doc, "3. Алгоритмуудын ажиллах зарчим ба харьцуулалт", level=1)
    algo_rows = [
        ["Boosting", "Эхний болон одоогийн хүрээний доторх Haar/интеграл шинж чанарыг ажиглана.",
         "Онлайн AdaBoost ангилагчдын жинг шинэчилж, эргэн тойронд хамгийн өндөр оноотой байрлалыг хайна.",
         "Сонгосон цорын ганц шилдэг тэгш өнцөгт (тогтмол хэмжээтэй), success дохио.",
         "Алдаатай дагасан хүрээг шууд эерэг жишээ болгон суралцдаг тул бага зэрэг хазайхад л дэвсгэр эсвэл баганад наалдаж гацдаг."],
        ["MIL", "Объектын эргэн тойрны жижиг шилжилттэй олон дэд хүрээний багц (bag)-аас Haar шинж чанар ажиглана.",
         "Multiple Instance Learning-ээр багц дотор ядаж нэг зөв хүрээ байхаар ангилагчийг шинэчилнэ.",
         "Оновчтой тэгш өнцөгт (тогтмол хэмжээтэй), success дохио.",
         "Багана бүрэн халхалсан ч success=True гэж буцаан багана дээр гацсан (ctrl_bicycle_occ дээр 5/5 буруу эерэг); хэмжээ өөрчлөгдөхөд дасахгүй."],
        ["KCF", "ROI ба орчны томруулсан цонхноос HOG/өнгөний шинж чанарыг мөчлөгт шилжилтээр ажиглана.",
         "Фурье огторгуйд (FFT) цөмийн корреляцийн шүүлтүүрийн оргил хариу ба итгэлцлийг бодно.",
         "Шилжсэн тэгш өнцөгт (тогтмол хэмжээтэй) ба найдвартай success дохио.",
         "Хэмжээ огцом өөрчлөгдөх (scale), хурдан хөдөлгөөн эсвэл халхлалтын үед оргил хариу унаж success=False болдог ба эргэн сэргэх детекторгүй."],
        ["TLD", "Саарал дүрсний 10×10 цэгийн хөдөлгөөн болон анхнаас хойш цуглуулсан объектын загварыг ажиглана.",
         "Track (MedianFlow), Detect (бүх кадрын скан тор + ангилагч), Learn (P/N эксперт шинэчлэл) гурвыг хослуулна.",
         "Масштаб нь өөрчлөгдөх тэгш өнцөгт ба итгэлгүй үед success=False.",
         "Бүрэн халхлалтын дараа дахин илрүүлдэг давуу талтай ч дэвсгэрт төстэй бараан объект таарахад буруу үсэрч болно (bmx-trees frame 72)."],
        ["MedianFlow", "ROI доторх 10×10 жигд торын цэгүүдийн орчныг хоёр дараалсан кадрт ажиглана.",
         "LK урсгалаар цэг бүрийг урагш-буцааж шалгаад (FB + NCC), үлдсэн цэгүүдийн медиан шилжилт ба хос зайн масштабыг тооцно.",
         "Нарийвчлалтай байрлал ба масштаб өөрчлөгдсөн тэгш өнцөгт.",
         "Объектын >50% халхлагдах эсвэл хүчтэй хөдөлгөөний бүрсийлт (motion blur) гарвал цэгүүдийн тал хувь эвдэрч шууд алддаг (bmx-trees IoU=0.0)."],
        ["CSRT", "Объектын дотор ба гадна өнгөний гистограм (сегментацийн маск) + HOG + Colornames сувгуудыг ажиглана.",
         "Орон зайн найдвартай байдлын маскаар зөвхөн объектод хамаарах хэсгийг жигнэн DCF шүүлтүүр ба DSST масштаб бодно.",
         "Масштаб ба чиглэлийн нарийвчлалтай тэгш өнцөгт, success дохио.",
         "Удаавтар ажилладаг (39–62 FPS, 1 thread); бүрэн халхлалтын дараа глобал детекторгүй тул өөрөө эргэж олохгүй."],
        ["Lucas–Kanade (LK)", "Shi–Tomasi-аар сонгосон цөөн булангийн цэгийн орчны (winSize) тод байдлын градиентыг ажиглана.",
         "Гэрэлтүүлэг тогтмол ба цонх доторх пикселүүд ижил шилжинэ гэж үзээд пирамид түвшинд (u, v) тэгшитгэл бодно.",
         "Цэг бүрийн бутархай шилжилт (x_new, y_new) ба status.",
         "Хэт том цонх (200×200) нь объектын ирмэг дээр хөдөлгөөнт объект ба зогсоо дэвсгэрийг хольж алдаа гаргана; хэт жижиг цонх хурдан үсрэлтийг гүйцэхгүй."],
        ["Farnebäck (dense)", "Дүрсний пиксел бүрийн орчны тод байдлыг ажиглана.",
         "Орчныг 2 зэрэгтийн олон гишүүнтээр (f(x) ≈ x^T A x + b^T x + c) ойролцуулж, пирамид түвшинд коэффициент шилжилтээс (u, v) тооцно.",
         "Пиксел бүрийн 2D вектор талбай (H×W×2) ба HSV өнгөний зураг.",
         "Сийрэг LK-аас ~15–150 дахин удаан (35–48 ms/frame); бүтэцгүй гөлгөр хэсэг болон объектын хурц ирмэг дээр бага зэрэг зөөлрүүлдэг."],
    ]
    add_table(
        doc,
        ["Алгоритм", "Юу ажигладаг", "Дараагийн кадрт юу тооцдог", "Юу гаргадаг", "Ямар нөхцөлд алдсан"],
        algo_rows,
        "Хүснэгт 2. Найман алгоритмын ажиллах зарчим ба алдааны нөхцөл.",
        font_pt=8,
    )
    add_p(
        doc,
        "Ойлгоход хялбар холбоос (дугуйн жишээгээр): "
        "Boosting нь дугуй бага зэрэг хазайсан хүрээг 'яг энэ дугуй мөн' гэж шууд сурдаг тул баганын ирмэг рүү хальтирдаг бол MIL нь эргэн тойрны хэдэн хүрээг багцаар шалгаж "
        "арай тэсвэртэй боловч зогссон баганыг дугуй гэж андуурсаар байдаг. KCF нь тэгш өнцөгт бүхэл цонхны доторх хэвийг FFT-ээр хурдан тулгадаг бол CSRT нь дугуйг дэвсгэрээс нь "
        "ялгасан маск (spatial reliability) болон суваг бүрийн жингээр зөвхөн дугуйны пикселийг онцолж, хэмжээг нь хамт тооцдог тул модны цаанаас цухуйх BMX дээр илүү гарчээ. "
        "MedianFlow нь дугуй дээр 100 цэг тавиад урагш-буцааж шалгаад дундаж тэлэлт/агшилтыг авдаг тул цэвэрхэн жижгэрэх машин (car-shadow) дээр 847 FPS-ээр тэргүүлсэн бол "
        "TLD нь MedianFlow дээр бүх дэлгэцээр хайдаг детектор ба P/N сургагчийг залгаснаар дугуй баганын цаана 5 кадр бүтэн алга болоод гарахад дахин олж чадсан ганц арга болов. "
        "LK нь булан цэгийн жижиг цонхоор тод байдлын налууг дагадаг бол Farnebäck нь пиксел бүрийн орчныг парабол гадаргуу (2D полином) болгож түүний шилжилтээр бүх пикселийн хөдөлгөөнийг гаргадаг."
    )

    # =========================================================================
    # 4. ХЭРЭГЛЭСЭН ӨГӨГДӨЛ БА ШУДАРГА ХЭМЖИЛТИЙН НӨХЦӨЛ
    # =========================================================================
    add_h(doc, "4. Хэрэглэсэн өгөгдөл ба шударга хэмжилтийн нөхцөл", level=1)
    add_p(
        doc,
        "Бид туршилтдаа 3 төрлийн өгөгдөл ашиглав: (1) DAVIS 2017-ийн бодит видеонууд (bmx-trees: 80 кадр, модны ар дахь хүнд халхлалт ба хэмжээний агшилт; "
        "car-shadow: 40 кадр, сүүдрийн өөрчлөлт ба ~3.5 дахин жижгэрэлт — объектын ID-аар сегментацийн маскнаас гаргасан тул эдгээр GT хүрээ нь бүтэн объектынх биш, "
        "харагдаж буй хэсгийнх (visible-part GT) юм); (2) Middlebury public GT хос зургууд (RubberWhale, Grove2, frame10→frame11, |u|,|v|>1e9 буюу халхлагдсан пикселийг GT маскнаас хассан); "
        "(3) Wikimedia Commons-ийн CC0 1.0 Front_Side_of_Bicyle.jpg фото зургаас үүсгэсэн фотогоор үүсгэсэн хяналттай туршилтууд (ctrl_bicycle_occ: 40 кадр, 18–22-р кадрт 100% бүтэн халхлалт + хэмжээний өөрчлөлт + төстэй саад объект; "
        "ctrl_bicycle_cam_shift: 24 кадр бүхэл кадрын (−6, −1) шилжилт; мөн obj_occ ба cam_shift хос зураг дээр шинээр ил гарсан, кадраас гарсан болон халхлагдсан пикселийг flow GT маскаас хассан)."
    )
    m_rows = [
        [r["dataset_id"], r["frames_used"], r["license_or_terms"], r["sha256"][:16] + "..."]
        for r in manifest
    ]
    add_table(
        doc,
        ["Өгөгдлийн ID", "Кадрын муж", "Лиценз / нөхцөл", "SHA-256 (эхний 16 тэмдэгт)"],
        m_rows,
        "Хүснэгт 3. Хэрэглэсэн эх өгөгдлийн бүртгэл (бүтэн SHA-256 ба URL-ууд results/csv/dataset_manifest.csv-д бий). Код Apache-2.0, фото CC0, DAVIS CC BY-NC 4.0 тул датасетийн том архивыг GitHub-д оруулаагүй.",
        font_pt=8,
        col_widths=[2.10, 1.55, 2.15, 1.40],
    )
    add_p(
        doc,
        "Шударга хэмжилтийн нөхцөл: Бүх туршилтыг ижил CPU (Intel Core 5 120U), ижил ганц урсгал (cv2.setNumThreads(1), OMP/OPENBLAS/MKL=1), ижил нягтрал дээр гүйцэтгэв. "
        "Файл унших, зураг зурах, CSV бичих хугацааг тусгаарлаж зөвхөн алгоритмын update()/flow тооцооллын 3 давталтын медиан ms/frame-ийг авсан. "
        "Эхний кадрыг (frame 0) оноонд оруулаагүй, дундаас GT-ээр дахин эхлүүлээгүй. GT-тэй кадрт tracker эвдрэл мэдэгдвэл (success=False) эсвэл хүчингүй хүрээ гаргавал IoU=0.0 оноов. "
        "Сэргэх хугацааны босгыг туршилтаас өмнө 'дахин харагдсанаас хойш IoU ≥ 0.5 дараалсан 3 кадр' гэж тогтоосон."
    )

    # =========================================================================
    # 5. БОДИТ ХЭМЖИЛТИЙН ДҮН
    # =========================================================================
    add_h(doc, "5. Бодит хэмжилтийн дүн", level=1)
    add_p(doc, "5.1 Объект дагах (Object Tracking) харьцуулалт", bold=True)
    t_rows = []
    for r in track_summary:
        t_rows.append([
            r["sequence"], r["tracker"], r["eval_frames"], r["absent_frames"],
            r["mean_iou"], r["iou_ge_05_pct"] + "%", r["failure_pct"] + "%",
            r["drift_success_true"], r["absent_fp_pct"] + "%", r["recovery_frames"],
            r["median_ms"], r["fps"],
        ])
    add_table(
        doc,
        ["Дараалал", "Tracker", "Үнэлэх", "Алга", "Mean IoU", "IoU≥0.5", "Эвдрэл%", "Drift(T)", "Алга FP%", "Сэргэлт", "ms/fr", "FPS"],
        t_rows,
        "Хүснэгт 4. Бүх 5 дарааллын хэмжилт (results/csv/track_summary.csv). Тэмдэглэл: STATIC нь ямар ч тооцоо хийхгүй анхны хүрээг буцаадаг (<0.001 ms) тул FPS нь хоосон суурь утга.",
        font_pt=7,
        col_widths=[1.38, 0.85, 0.42, 0.36, 0.54, 0.55, 0.55, 0.48, 0.53, 0.62, 0.44, 0.48],
    )

    add_p(doc, "5.2 Optical Flow (LK ба Farnebäck) харьцуулалт ба Ablation", bold=True)
    f_rows = [
        [r["dataset"], r["method"], f"{r['compared_pts']}/{r['gt_valid_pts']}", r["point_loss_pct"] + "%",
         r["mean_epe_at_pts"], r["mean_epe_all_px"], str(r["valid_px_count"]), r["median_ms"], r["fps"]]
        for r in flow_summary if r["max_corners"] == "10"
    ]
    add_table(
        doc,
        ["Өгөгдөл", "Арга", "Харьц. цэг", "Алдалт%", "EPE (цэг)", "EPE (бүх px)", "Хүчинтэй px", "ms/fr", "FPS"],
        f_rows,
        "Хүснэгт 5. Shi–Tomasi maxCorners=10 цэг дээрх LK ба Farnebäck харьцуулалт (results/csv/flow_summary.csv; maxCorners=100 өргөтгөл мөн CSV-д хадгалагдсан).",
        font_pt=8,
        col_widths=[1.42, 1.15, 0.62, 0.56, 0.68, 0.72, 0.75, 0.62, 0.68],
    )

    ab_rows = [
        [r["sequence"], r["variant"], r["winSize"], r["fb_check"], r["redetect_interval"],
         f"{r['final_pts']}/{r['init_pts']} ({r['retention_pct']}%)", r["mean_epe_px"], r["median_ms"], r["fps"]]
        for r in flow_ablation
    ]
    add_table(
        doc,
        ["Дараалал", "Туршилтын хувилбар", "winSize", "FB", "Redet", "Үлдсэн цэг", "EPE", "ms/fr", "FPS"],
        ab_rows,
        "Хүснэгт 6. LK урагш-буцах (FB) шалгалт, дахин илрүүлэлт болон winSize=(200,200) vs (15,15) тусгаарласан ablation (results/csv/flow_ablation.csv).",
        font_pt=7,
        col_widths=[1.48, 1.65, 0.72, 0.28, 0.42, 0.92, 0.55, 0.56, 0.62],
    )

    add_img(doc, FIGURES_DIR / "tracking_flow_metrics.png", 6.5,
            "Зураг 1. Объект дагах (Mean IoU, FPS) болон Optical Flow (EPE, LK FB/redetect retention) хэмжилтийн нэгдсэн график.")
    add_img(doc, FIGURES_DIR / "lk_vs_farneback_cc0.png", 6.5,
            "Зураг 2. CC0 хяналттай хос зураг дээрх сийрэг LK (улаан=win200, ногоон=win15) ба нягт Farnebäck (сумт тор ба тогтмол max=12px HSV өнгөний дугуйтай) дүрслэл.")

    # =========================================================================
    # 6. ЭВДРЭЛИЙН ЗУРАГТ ЖИШЭЭ БА ШИНЖИЛГЭЭ
    # =========================================================================
    add_h(doc, "6. Эвдрэлийн зурагт жишээ: «Дугуй баганын цаагуур ороход» ба бодит видео", level=1)
    add_img(doc, FIGURES_DIR / "bicycle_storyboard_cc0.png", 6.5,
            "Зураг 3. Хувилбар A (CC0 дугуйн объект баганын цаагуур орж гарах, Frame 4, 13, 19, 32): Мөр 1=GT, Мөр 2=CSRT/KCF/TLD, Мөр 3=LK цэгүүд, Мөр 4=Farnebäck HSV+сум.")
    add_img(doc, FIGURES_DIR / "camera_shift_storyboard_cc0.png", 6.5,
            "Зураг 4. Хувилбар B (CC0 дээрх бүхэл кадрын камерын шилжилт, Frame 1, 7, 14, 21): Мөр 4-т дэлгэц бүхэлдээ ижил чиглэлтэй (180° цэнхэр) урсгал үүсэж буйг Хувилбар A-тай харьцуулна уу.")
    add_img(doc, LOCAL_VIZ_DIR / "davis_bmx_failures.png", 6.2,
            "Зураг 5. Бодит DAVIS 2017 bmx-trees (obj_id=1 дугуй, CC BY-NC 4.0) дээрх эвдрэлийн кадрууд: Frame 10, 30, 61 (модны халхлалт), 72 (TLD хогийн сав руу үсэрсэн).")

    # Concrete frame analysis citing exact CSV numbers
    c_tld = tlookup("ctrl_bicycle_occ", "TLD")
    c_csrt = tlookup("ctrl_bicycle_occ", "CSRT")
    c_mil = tlookup("ctrl_bicycle_occ", "MIL")
    b_csrt = tlookup("davis_bmx_trees_bike", "CSRT")
    b_mf = tlookup("davis_bmx_trees_bike", "MEDIANFLOW")
    m_cs = tlookup("davis_car_shadow", "MEDIANFLOW")
    add_p(doc, "Сайн ба муу кадрын тоон баталгаа:", bold=True)
    add_bullets(doc, [
        f"Хувилбар A — Дугуй баганын цаагуур ороход (Зураг 3): Frame 4 (цэвэр хөдөлгөөн) дээр CSRT IoU=0.92, KCF IoU=0.62, TLD IoU=0.64 байв. Frame 13 (50% халхлалт) дээр KCF IoU=0.53, CSRT IoU=0.49 болж буурав. Frame 19 (100% халхлагдсан, алга кадр) дээр CSRT, KCF, TLD гурав нь бүгд success=False мэдэгдэж алга үеийн FP=0.0% байсан бол MIL, BOOSTING, MEDIANFLOW нь success=True гэж баганан дээр наалдсан (алга үеийн FP=100.0%). Frame 32 (дахин гарсны дараа) дээр TLD нь глобал детектороороо дугуйг IoU=0.85 болгон эргэн барьж (нийт сэргэх хугацаа {c_tld['recovery_frames']} кадр, mean IoU={c_tld['mean_iou']}), харин CSRT (fail={c_csrt['failure_pct']}%) ба KCF сэргэж чадаагүй.",
        "Хувилбар A ба B-ийн хөдөлгөөний сумны ялгаа (Зураг 3 vs Зураг 4): Хувилбар A-д зөвхөн хөдөлж буй дугуй (баруун 0°, улбар шар-улаан) ба саад объект (зүүн 180°, цэнхэр) дээр сум/өнгө үүсэж, Frame 19-д дугуй баганын цаана ороход тэр хэсгийн flow бүрэн хар (0 px) болов. Харин камерын хөдөлгөөн дуурайлгасан Хувилбар B-д дүрсний бүх пиксел ижил (−6, −1) шилжих тул дэлгэц бүхэлдээ цэнхэр өнгө, жигд зүүн сумаар дүүрч, бүх 6 tracker IoU≥0.5-ийг 100% хангав.",
        f"Бодит DAVIS bmx-trees (Зураг 5): Frame 10-д CSRT (IoU=0.97), MIL (0.94), KCF (0.79) сайн барьж байсан ч Frame 30-д хурдан хөдөлгөөн ба модны эхлэл дээр MIL болон TLD (success=True, IoU=0.00) зүүн талын мод руу хазайсан. Frame 61 (хүнд халхлалт, маск ердөө 426 px) ба Frame 72-т зөвхөн CSRT (mean IoU={b_csrt['mean_iou']}, 53.16%) дугуйг гүйцэж барьсан бол MedianFlow бүр эхнээсээ эвдэрч mean IoU={b_mf['mean_iou']} болов.",
        f"Бодит DAVIS car-shadow: Халхлалтгүй боловч 3.5 дахин жижгэрэх машинд MedianFlow нь pairwise scale тооцоороо mean IoU={m_cs['mean_iou']} (100% IoU≥0.5, {m_cs['fps']} FPS)-ээр тэргүүлж, CSRT (0.6612) удаалав.",
        f"Optical Flow цонх ба FB шүүлт: Controlled:obj_occ дээр LK winSize=(200,200) нь {flookup('Controlled:obj_occ','LK_PDF_win200')['mean_epe_at_pts']} px алдаа гаргасан (том цонх объектын хил дээр хөдөлгөөнгүй баганыг хамруулснаас) бол winSize=(15,15) нь {flookup('Controlled:obj_occ','LK_small_win15')['mean_epe_at_pts']} px болж ~20 дахин нарийссан ба ~10 дахин хурдан ({flookup('Controlled:obj_occ','LK_small_win15')['median_ms']} ms vs {flookup('Controlled:obj_occ','LK_PDF_win200')['median_ms']} ms) ажиллав. Мөн bicycle_occ_40f дээр FB шалгалтгүй LK нь бүх 10 цэгийг (100%) хадгалсан мэт харагдавч халхлагдсан цэгүүдийг дагуулаад алдсан бол FB шалгалт ({alookup('Controlled:bicycle_occ_40f','LK_PDF_win200_FB')['retention_pct']}%) буруу цэгүүдийг цагт нь хасаж, зөвхөн re-detect нэмэхэд 90% цэг сэргэсэн.",
    ])

    # =========================================================================
    # 7. ДҮГНЭЛТ БА БОДИТ ТӨЛӨВ
    # =========================================================================
    add_h(doc, "7. Дүгнэлт ба гүйцэтгэлийн бодит төлөв", level=1)
    add_bullets(doc, [
        "Аль бичлэг ба нөхцөлд аль арга давуу вэ: (1) Хэсэгчилсэн халхлалт, өнгө-хэлбэрийн өөрчлөлттэй бодит бичлэгт (bmx-trees) CSRT (mean IoU=0.5400–0.5914) хамгийн найдвартай; (2) Бүрэн халхлагдаад дахин гарч ирдэг үед (ctrl_bicycle_occ) глобал детектортой TLD (mean IoU=0.5057, 6 кадрт сэргэсэн) цорын ганц сэргээгч болов; (3) Халхлалтгүй, жигд жижгэрэх хатуу биетэд (car-shadow) MedianFlow (0.7271 IoU, 846.82 FPS) хурд, нарийвчлалаар аль алинаар нь тэргүүлэв; (4) Optical flow-д жижиг цонхтой LK (winSize=(15,15), EPE=0.099–0.239 px, ~0.3 ms) цөөн цэгт хамгийн хурдан бөгөөд хил дээр цэвэр, харин нягт хөдөлгөөний зураглалд Farnebäck (EPE=0.256–0.585 px, 37–48 ms) шаардлагатай.",
        "Туршилтын гүйцэтгэлийн төлөв: Бүх 6 tracker + STATIC суурь арга + LK + Farnebäck бүрэн ажилласан (BLOCKED болсон tracker үгүй — учир нь cv2 ба cv2.legacy аль алийг нь зөв илрүүлж нэг contrib wheel дээр баталгаажуулсан). Middlebury-ийн нууцлалтай eval багцын GT нийтэд хаалттай тул public GT бүхий RubberWhale, Grove2 хосыг, DAVIS 2017-г бүтэн 832 MB таталгүйгээр HTTP Range-ээр яг bmx-trees, car-shadow дарааллаар татаж ашигласан.",
        f"GitHub ба баталгаажуулалт: Цэвэр .venv орчинд unittest (7/7), README командаар давталт ба benchmark бүрэн шалгагдсан. Зөвхөн код, CC0 демо, хэмжилтийн CSV ба тайлан орсон цэвэр branch-ийг {repo_url_note} руу push хийсэн (сургалтын PDF болон датасетийн том архивыг оруулаагүй).",
    ])

    out_path = ROOT / "Tracking_Lab_Report_MN.docx"
    doc.save(out_path)
    return out_path


if __name__ == "__main__":
    p = build_report()
    print("Built report:", p)
