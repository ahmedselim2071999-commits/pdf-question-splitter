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


os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    RENDER_DIR,
    exist_ok=True
)

os.makedirs(
    CROP_DIR,
    exist_ok=True
)

os.makedirs(
    PPT_DIR,
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
# DETECT MAIN QUESTION NUMBER BOXES
# =========================================================

def detect_question_boxes(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:
        return []


    height, width = image.shape[:2]


    # -----------------------------------------------------
    # Convert to HSV
    # -----------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    # -----------------------------------------------------
    # RED COLOR RANGE
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
    # MORPHOLOGICAL CLEANUP
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


        box_area = w * h


        if box_area <= 0:
            continue


        contour_area = cv2.contourArea(
            contour
        )


        fill_ratio = (
            contour_area /
            float(box_area)
        )


        aspect_ratio = (
            w /
            float(h)
        )


        # -------------------------------------------------
        # SIZE FILTER
        # -------------------------------------------------

        if w < 25 or h < 25:
            continue


        if w > 120 or h > 120:
            continue


        # -------------------------------------------------
        # SQUARE FILTER
        # -------------------------------------------------

        if aspect_ratio < 0.70:
            continue


        if aspect_ratio > 1.40:
            continue


        # -------------------------------------------------
        # FILLED BOX FILTER
        # -------------------------------------------------

        if fill_ratio < 0.55:
            continue


        # -------------------------------------------------
        # MAIN QUESTION BOXES ARE ON LEFT SIDE
        # -------------------------------------------------

        if x > width * 0.25:
            continue


        # -------------------------------------------------
        # IGNORE TOP HEADER
        # -------------------------------------------------

        if y < height * 0.15:
            continue


        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": box_area,
            "fill_ratio": fill_ratio
        })


    # -----------------------------------------------------
    # SORT TOP → BOTTOM
    # -----------------------------------------------------

    candidates.sort(
        key=lambda item: item["y"]
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


    # -----------------------------------------------------
    # FINAL SORT
    # -----------------------------------------------------

    questions.sort(
        key=lambda item: item["y"]
    )


    return questions


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
        # START
        # -------------------------------------------------

        top = max(
            0,
            question["y"] - 25
        )


        # -------------------------------------------------
        # END
        # -------------------------------------------------

        if index < len(questions) - 1:

            next_question = (
                questions[index + 1]
            )


            bottom = max(
                top + 100,
                next_question["y"] - 20
            )

        else:

            bottom = height - 25


        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------

        if bottom <= top:
            continue


        # -------------------------------------------------
        # CROP
        # -------------------------------------------------

        crop = image[
            top:bottom,
            0:width
        ]


        # -------------------------------------------------
        # FILE NAME
        # -------------------------------------------------

        crop_path = os.path.join(
            CROP_DIR,
            f"page_{page_number}_question_{index + 1}.png"
        )


        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

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

    # -----------------------------------------------------
    # CREATE PRESENTATION
    # -----------------------------------------------------

    prs = Presentation()


    # -----------------------------------------------------
    # 16:9 WIDESCREEN
    # -----------------------------------------------------

    prs.slide_width = Inches(
        13.333
    )

    prs.slide_height = Inches(
        7.5
    )


    # -----------------------------------------------------
    # CREATE ONE SLIDE PER QUESTION
    # -----------------------------------------------------

    for question in questions:

        # -------------------------------------------------
        # BLANK SLIDE
        # -------------------------------------------------

        slide = prs.slides.add_slide(
            prs.slide_layouts[6]
        )


        # -------------------------------------------------
        # IMAGE
        # -------------------------------------------------

        image_path = question["path"]


        if not os.path.exists(
            image_path
        ):
            continue


        # -------------------------------------------------
        # GET IMAGE SIZE
        # -------------------------------------------------

        image = cv2.imread(
            image_path
        )


        if image is None:
            continue


        image_height, image_width = (
            image.shape[:2]
        )


        # -------------------------------------------------
        # POWERPOINT DIMENSIONS
        # -------------------------------------------------

        slide_width = (
            prs.slide_width
        )

        slide_height = (
            prs.slide_height
        )


        # -------------------------------------------------
        # SMALL MARGIN
        # -------------------------------------------------

        margin = Inches(
            0.15
        )


        available_width = (
            slide_width -
            (margin * 2)
        )


        available_height = (
            slide_height -
            (margin * 2)
        )


        # -------------------------------------------------
        # IMAGE ASPECT RATIO
        # -------------------------------------------------

        image_ratio = (
            image_width /
            float(image_height)
        )


        slide_ratio = (
            available_width /
            float(available_height)
        )


        # -------------------------------------------------
        # FIT IMAGE INSIDE SLIDE
        # -------------------------------------------------

        if image_ratio > slide_ratio:

            # Image is wider

            final_width = (
                available_width
            )

            final_height = (
                int(
                    available_width /
                    image_ratio
                )
            )

        else:

            # Image is taller

            final_height = (
                available_height
            )

            final_width = (
                int(
                    available_height *
                    image_ratio
                )
            )


        # -------------------------------------------------
        # CENTER IMAGE
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
    # SAVE POWERPOINT
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
    # CHECK FILE
    # -----------------------------------------------------

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        return """
        <h2>❌ Please upload a PDF file.</h2>
        """


    # -----------------------------------------------------
    # CREATE SAFE FILE NAME
    # -----------------------------------------------------

    safe_name = (
        os.path.basename(
            file.filename
        )
    )


    file_path = os.path.join(
        UPLOAD_DIR,
        safe_name
    )


    # -----------------------------------------------------
    # SAVE PDF
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
    # PROCESS EVERY PAGE
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
        # ADD TO ALL QUESTIONS
        # -------------------------------------------------

        for crop in crops:

            all_questions.append(
                crop
            )


    pdf.close()


    # =====================================================
    # CREATE POWERPOINT
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

            .download:hover {{
                background: #15803d;
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


    # -----------------------------------------------------
    # POWERPOINT DOWNLOAD
    # -----------------------------------------------------

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
    # SHOW QUESTIONS
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
# SERVE CROP IMAGE
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
# DOWNLOAD POWERPOINT
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
