from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import fitz
import os
import cv2
import numpy as np


app = FastAPI(title="PDF Question Splitter")


# =========================================================
# DIRECTORIES
# =========================================================

UPLOAD_DIR = "uploads"
RENDER_DIR = "rendered"
CROP_DIR = "crops"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RENDER_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
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
                padding: 12px 25px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
            }

            button:hover {
                background: #1d4ed8;
            }

        </style>

    </head>


    <body>

        <div class="box">

            <h1>📚 PDF Question Splitter</h1>

            <p>
                Upload a PDF and automatically split
                the main questions.
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
                    Analyze PDF
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

    image = cv2.imread(image_path)

    if image is None:
        return []


    height, width = image.shape[:2]


    # -----------------------------------------------------
    # Convert image to HSV
    # -----------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    # -----------------------------------------------------
    # RED COLOR RANGES
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
    # CLOSE SMALL HOLES
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
    # FIND RED OBJECTS
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
        # DON'T DETECT HEADER
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
    questions
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
        # TOP OF QUESTION
        # -------------------------------------------------

        top = max(
            0,
            question["y"] - 25
        )


        # -------------------------------------------------
        # BOTTOM OF QUESTION
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
        # CREATE CROP
        # -------------------------------------------------

        crop = image[
            top:bottom,
            0:width
        ]


        # -------------------------------------------------
        # SAVE CROP
        # -------------------------------------------------

        crop_path = os.path.join(
            CROP_DIR,
            f"question_{index + 1}.png"
        )


        cv2.imwrite(
            crop_path,
            crop
        )


        crops.append({
            "number": index + 1,
            "path": crop_path
        })


    return crops


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
    # SAVE PDF
    # -----------------------------------------------------

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )


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


    page_count = len(pdf)


    all_questions = []


    # =====================================================
    # PROCESS EVERY PAGE
    # =====================================================

    for page_index in range(
        page_count
    ):


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
            f"page_{page_index + 1}.png"
        )


        pix.save(
            page_path
        )


        # -------------------------------------------------
        # DETECT QUESTION BOXES
        # -------------------------------------------------

        questions = detect_question_boxes(
            page_path
        )


        # -------------------------------------------------
        # CROP QUESTIONS
        # -------------------------------------------------

        crops = crop_questions(
            page_path,
            questions
        )


        # -------------------------------------------------
        # ADD PAGE NUMBER
        # -------------------------------------------------

        for crop in crops:

            crop["page"] = (
                page_index + 1
            )


            all_questions.append(
                crop
            )


    pdf.close()


    # =====================================================
    # RESULT PAGE
    # =====================================================

    html = f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>Question Detection</title>

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
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
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

            .number {{
                color: #2563eb;
                font-weight: bold;
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

            </div>
    """


    # -----------------------------------------------------
    # NO QUESTIONS
    # -----------------------------------------------------

    if not all_questions:

        html += """

            <div class="question">

                <h2>
                    ⚠️ No questions detected
                </h2>

                <p>
                    The system could not detect
                    the main question markers.
                </p>

            </div>

        """


    # -----------------------------------------------------
    # SHOW QUESTIONS
    # -----------------------------------------------------

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
# SERVE CROPPED IMAGE
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
