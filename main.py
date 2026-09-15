from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse

import fitz
import os
import cv2
import numpy as np

from pptx import Presentation
from pptx.util import Inches


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="PDF Question Splitter"
)


# =========================================================
# DIRECTORIES
# =========================================================

UPLOAD_DIR = "uploads"
RENDER_DIR = "rendered"
CROP_DIR = "crops"
PPT_DIR = "powerpoints"


for folder in [
    UPLOAD_DIR,
    RENDER_DIR,
    CROP_DIR,
    PPT_DIR
]:
    os.makedirs(
        folder,
        exist_ok=True
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>PDF Question Splitter</title>

        <style>

            body {
                font-family: Arial, sans-serif;
                background: #f4f6f8;
                text-align: center;
                padding: 50px;
            }

            .box {
                background: white;
                max-width: 900px;
                margin: auto;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 5px 25px rgba(0,0,0,0.1);
            }

            h1 {
                margin-bottom: 10px;
            }

            p {
                color: #666;
            }

            input {
                margin: 25px 0;
            }

            button {
                background: #2563eb;
                color: white;
                border: none;
                padding: 14px 28px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 17px;
            }

            button:hover {
                background: #1d4ed8;
            }

        </style>

    </head>


    <body>

        <div class="box">

            <h1>
                📚 PDF Question Splitter
            </h1>

            <p>
                Upload a PDF containing questions
                and automatically create a PowerPoint.
            </p>


            <form
                action="/upload"
                method="post"
                enctype="multipart/form-data"
            >

                <input
                    type="file"
                    name="file"
                    accept=".pdf"
                    required
                >

                <br><br>

                <button type="submit">
                    🚀 Analyze PDF
                </button>

            </form>

        </div>

    </body>

    </html>
    """


# =========================================================
# DETECT QUESTION NUMBER BOXES
# =========================================================

def detect_question_boxes(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:
        return []


    height, width = image.shape[:2]


    # -----------------------------------------------------
    # HSV
    # -----------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    # -----------------------------------------------------
    # RED MASK
    # -----------------------------------------------------

    lower_red_1 = np.array(
        [0, 100, 80]
    )

    upper_red_1 = np.array(
        [12, 255, 255]
    )


    lower_red_2 = np.array(
        [168, 100, 80]
    )

    upper_red_2 = np.array(
        [180, 255, 255]
    )


    mask1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1
    )


    mask2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2
    )


    red_mask = mask1 | mask2


    # -----------------------------------------------------
    # CLEAN MASK
    # -----------------------------------------------------

    kernel = np.ones(
        (5, 5),
        np.uint8
    )


    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # -----------------------------------------------------
    # FIND CONTOURS
    # -----------------------------------------------------

    contours, _ = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    candidates = []


    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )


        area = w * h


        if area <= 0:
            continue


        contour_area = cv2.contourArea(
            contour
        )


        fill_ratio = (
            contour_area /
            float(area)
        )


        aspect_ratio = (
            w /
            float(h)
        )


        # -------------------------------------------------
        # SIZE
        # -------------------------------------------------

        if w < 25 or h < 25:
            continue


        if w > 120 or h > 120:
            continue


        # -------------------------------------------------
        # SHAPE
        # -------------------------------------------------

        if aspect_ratio < 0.70:
            continue


        if aspect_ratio > 1.40:
            continue


        # -------------------------------------------------
        # FILLED RED BOX
        # -------------------------------------------------

        if fill_ratio < 0.55:
            continue


        # -------------------------------------------------
        # LEFT SIDE
        # -------------------------------------------------

        if x > width * 0.25:
            continue


        # -------------------------------------------------
        # IGNORE HEADER
        # -------------------------------------------------

        if y < height * 0.15:
            continue


        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })


    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    candidates.sort(
        key=lambda q: q["y"]
    )


    # -----------------------------------------------------
    # REMOVE DUPLICATES
    # -----------------------------------------------------

    questions = []


    for candidate in candidates:

        duplicate = False


        for existing in questions:

            if (
                abs(
                    candidate["x"] -
                    existing["x"]
                ) < 25
                and
                abs(
                    candidate["y"] -
                    existing["y"]
                ) < 25
            ):

                duplicate = True
                break


        if not duplicate:

            questions.append(
                candidate
            )


    return questions


# =========================================================
# SMART CONTENT BOUNDARY
# =========================================================

def find_content_bounds(
    crop
):

    if crop is None:
        return crop


    if crop.size == 0:
        return crop


    # -----------------------------------------------------
    # GRAYSCALE
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )


    # -----------------------------------------------------
    # DARK PIXELS
    #
    # Ignore very light watermark/background
    # -----------------------------------------------------

    dark_mask = (
        gray < 220
    ).astype(
        np.uint8
    ) * 255


    # -----------------------------------------------------
    # REMOVE SMALL NOISE
    # -----------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )


    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        kernel
    )


    # -----------------------------------------------------
    # ROW ACTIVITY
    # -----------------------------------------------------

    row_counts = np.sum(
        dark_mask > 0,
        axis=1
    )


    col_counts = np.sum(
        dark_mask > 0,
        axis=0
    )


    # -----------------------------------------------------
    # FIND ACTIVE ROWS
    # -----------------------------------------------------

    min_row_pixels = max(
        5,
        int(crop.shape[1] * 0.001)
    )


    active_rows = np.where(
        row_counts > min_row_pixels
    )[0]


    if len(active_rows) == 0:
        return crop


    top = int(
        active_rows[0]
    )


    bottom = int(
        active_rows[-1]
    )


    # -----------------------------------------------------
    # FIND ACTIVE COLUMNS
    # -----------------------------------------------------

    min_col_pixels = max(
        5,
        int(crop.shape[0] * 0.001)
    )


    active_cols = np.where(
        col_counts > min_col_pixels
    )[0]


    if len(active_cols) > 0:

        left = int(
            active_cols[0]
        )

        right = int(
            active_cols[-1]
        )

    else:

        left = 0
        right = crop.shape[1] - 1


    # -----------------------------------------------------
    # SAFETY MARGIN
    # -----------------------------------------------------

    margin_x = 25
    margin_y = 25


    left = max(
        0,
        left - margin_x
    )


    right = min(
        crop.shape[1] - 1,
        right + margin_x
    )


    top = max(
        0,
        top - margin_y
    )


    bottom = min(
        crop.shape[0] - 1,
        bottom + margin_y
    )


    # -----------------------------------------------------
    # FINAL CROP
    # -----------------------------------------------------

    result = crop[
        top:bottom + 1,
        left:right + 1
    ]


    return result


# =========================================================
# CROP QUESTIONS
# =========================================================

def crop_questions(
    image_path,
    questions,
    page_number
):

    image = cv2.imread(
        image_path
    )


    if image is None:
        return []


    height, width = image.shape[:2]


    crops = []


    for index, question in enumerate(
        questions
    ):


        # -------------------------------------------------
        # TOP
        # -------------------------------------------------

        top = max(
            0,
            question["y"] - 30
        )


        # -------------------------------------------------
        # BOTTOM
        # -------------------------------------------------

        if index < len(questions) - 1:

            next_question = (
                questions[index + 1]
            )


            bottom = max(
                top + 100,
                next_question["y"] - 25
            )

        else:

            # ---------------------------------------------
            # Last question
            # Don't automatically use entire page
            # ---------------------------------------------

            bottom = height - 20


        # -------------------------------------------------
        # LEFT
        # -------------------------------------------------

        left = max(
            0,
            question["x"] - 30
        )


        # -------------------------------------------------
        # RIGHT
        #
        # Keep most of question area but remove
        # extreme right-side watermark area
        # -------------------------------------------------

        right = int(
            width * 0.94
        )


        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------

        if bottom <= top:
            continue


        if right <= left:
            continue


        # -------------------------------------------------
        # INITIAL CROP
        # -------------------------------------------------

        crop = image[
            top:bottom,
            left:right
        ]


        if crop.size == 0:
            continue


        # -------------------------------------------------
        # SMART TRIM
        # -------------------------------------------------

        crop = find_content_bounds(
            crop
        )


        if crop.size == 0:
            continue


        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        crop_path = os.path.join(
            CROP_DIR,
            (
                f"page_{page_number}"
                f"_question_{index + 1}.png"
            )
        )


        cv2.imwrite(
            crop_path,
            crop
        )


        crops.append({
            "number": index + 1,
            "page": page_number,
            "path": crop_path
        })


    return crops


# =========================================================
# CREATE POWERPOINT
# =========================================================

def create_powerpoint(
    questions,
    output_path
):

    prs = Presentation()


    # -----------------------------------------------------
    # 16:9
    # -----------------------------------------------------

    prs.slide_width = Inches(
        13.333
    )

    prs.slide_height = Inches(
        7.5
    )


    for question in questions:

        slide = prs.slides.add_slide(
            prs.slide_layouts[6]
        )


        image_path = question["path"]


        if not os.path.exists(
            image_path
        ):
            continue


        image = cv2.imread(
            image_path
        )


        if image is None:
            continue


        image_height, image_width = (
            image.shape[:2]
        )


        slide_width = (
            prs.slide_width
        )

        slide_height = (
            prs.slide_height
        )


        # -------------------------------------------------
        # SLIDE MARGIN
        # -------------------------------------------------

        margin = Inches(
            0.30
        )


        available_width = (
            slide_width -
            margin * 2
        )


        available_height = (
            slide_height -
            margin * 2
        )


        # -------------------------------------------------
        # IMAGE RATIO
        # -------------------------------------------------

        image_ratio = (
            image_width /
            float(image_height)
        )


        available_ratio = (
            available_width /
            float(available_height)
        )


        # -------------------------------------------------
        # FIT IMAGE
        # -------------------------------------------------

        if image_ratio > available_ratio:

            final_width = (
                available_width
            )

            final_height = int(
                available_width /
                image_ratio
            )

        else:

            final_height = (
                available_height
            )

            final_width = int(
                available_height *
                image_ratio
            )


        # -------------------------------------------------
        # CENTER
        # -------------------------------------------------

        left = int(
            (
                slide_width -
                final_width
            ) / 2
        )


        top = int(
            (
                slide_height -
                final_height
            ) / 2
        )


        # -------------------------------------------------
        # ADD IMAGE
        # -------------------------------------------------

        slide.shapes.add_picture(
            image_path,
            left,
            top,
            width=final_width,
            height=final_height
        )


    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    prs.save(
        output_path
    )


    return output_path


# =========================================================
# UPLOAD PDF
# =========================================================

@app.post(
    "/upload",
    response_class=HTMLResponse
)
async def upload_pdf(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # CHECK PDF
    # -----------------------------------------------------

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        return """
        <h2>❌ Please upload a PDF file.</h2>
        """


    # -----------------------------------------------------
    # SAFE NAME
    # -----------------------------------------------------

    safe_name = os.path.basename(
        file.filename
    )


    file_path = os.path.join(
        UPLOAD_DIR,
        safe_name
    )


    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    with open(
        file_path,
        "wb"
    ) as buffer:

        buffer.write(
            await file.read()
        )


    # -----------------------------------------------------
    # OPEN PDF
    # -----------------------------------------------------

    pdf = fitz.open(
        file_path
    )


    page_count = len(
        pdf
    )


    all_questions = []


    # =====================================================
    # PROCESS PAGES
    # =====================================================

    for page_index in range(
        page_count
    ):


        page_number = (
            page_index + 1
        )


        page = pdf[
            page_index
        ]


        # -------------------------------------------------
        # HIGH QUALITY RENDER
        # -------------------------------------------------

        matrix = fitz.Matrix(
            2.5,
            2.5
        )


        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )


        page_path = os.path.join(
            RENDER_DIR,
            f"page_{page_number}.png"
        )


        pix.save(
            page_path
        )


        # -------------------------------------------------
        # DETECT
        # -------------------------------------------------

        questions = detect_question_boxes(
            page_path
        )


        # -------------------------------------------------
        # CROP
        # -------------------------------------------------

        crops = crop_questions(
            page_path,
            questions,
            page_number
        )


        # -------------------------------------------------
        # STORE
        # -------------------------------------------------

        all_questions.extend(
            crops
        )


    pdf.close()


    # =====================================================
    # CREATE PPT
    # =====================================================

    ppt_filename = (
        os.path.splitext(
            safe_name
        )[0]
        +
        "_Split_Questions.pptx"
    )


    ppt_path = os.path.join(
        PPT_DIR,
        ppt_filename
    )


    if all_questions:

        create_powerpoint(
            all_questions,
            ppt_path
        )


    # =====================================================
    # RESULT PAGE
    # =====================================================

    html = f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>PDF Question Splitter</title>

        <style>

            body {{
                font-family: Arial, sans-serif;
                background: #f4f6f8;
                padding: 40px;
            }}

            .container {{
                max-width: 1100px;
                margin: auto;
                background: white;
                padding: 35px;
                border-radius: 15px;
                box-shadow: 0 5px 25px rgba(0,0,0,0.1);
            }}

            .summary {{
                background: #f0fdf4;
                padding: 25px;
                border-radius: 12px;
                margin-bottom: 30px;
            }}

            .download {{
                display: inline-block;
                background: #16a34a;
                color: white;
                text-decoration: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
                margin-top: 15px;
            }}

            .question {{
                margin-top: 35px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 12px;
                background: #fafafa;
            }}

            .question img {{
                max-width: 100%;
                border: 1px solid #ccc;
                margin-top: 15px;
            }}

        </style>

    </head>


    <body>

        <div class="container">

            <h1>
                🧠 Question Detection
            </h1>


            <div class="summary">

                <h2>
                    ✅ {len(all_questions)}
                    questions detected
                </h2>


                <p>
                    PDF pages:
                    <strong>
                        {page_count}
                    </strong>
                </p>
    """


    if all_questions:

        html += f"""

                <a
                    class="download"
                    href="/download/{ppt_filename}"
                >
                    ⬇️ Download PowerPoint
                </a>

        """

    else:

        html += """

                <p>
                    ⚠️ No questions detected.
                </p>

        """


    html += """

            </div>

    """


    # =====================================================
    # QUESTION PREVIEWS
    # =====================================================

    for index, question in enumerate(
        all_questions
    ):

        filename = os.path.basename(
            question["path"]
        )


        html += f"""

            <div class="question">

                <h2>
                    Question {index + 1}
                </h2>


                <p>
                    Page:
                    <strong>
                        {question["page"]}
                    </strong>
                </p>


                <img
                    src="/crop/{filename}"
                >

            </div>

        """


    html += """

        </div>

    </body>

    </html>
    """


    return html


# =========================================================
# SERVE CROP
# =========================================================

@app.get(
    "/crop/{filename}"
)
def get_crop(
    filename: str
):

    path = os.path.join(
        CROP_DIR,
        filename
    )


    if not os.path.exists(
        path
    ):

        return {
            "error":
            "Image not found."
        }


    return FileResponse(
        path,
        media_type="image/png"
    )


# =========================================================
# DOWNLOAD PPT
# =========================================================

@app.get(
    "/download/{filename}"
)
def download_powerpoint(
    filename: str
):

    path = os.path.join(
        PPT_DIR,
        filename
    )


    if not os.path.exists(
        path
    ):

        return {
            "error":
            "PowerPoint file not found."
        }


    return FileResponse(
        path,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "presentationml.presentation"
        ),
        filename=filename
    )
