from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import fitz
import os
import cv2
import numpy as np


app = FastAPI(title="PDF Question Splitter")


UPLOAD_DIR = "uploads"
RENDER_DIR = "rendered"
CROP_DIR = "crops"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RENDER_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)


# =========================================================
# HOME
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
                Upload a PDF and automatically detect individual questions.
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
# DETECT QUESTION NUMBER BOXES
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
    # Detect RED
    # -----------------------------------------------------

    lower_red_1 = np.array(
        [0, 80, 80]
    )

    upper_red_1 = np.array(
        [15, 255, 255]
    )


    lower_red_2 = np.array(
        [165, 80, 80]
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
    # Clean mask
    # -----------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )


    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # -----------------------------------------------------
    # Find contours
    # -----------------------------------------------------

    contours, _ = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    candidates = []


    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)


        area = w * h


        # -------------------------------------------------
        # Main question number boxes are relatively small.
        # -------------------------------------------------

        if area < 150:
            continue


        if area > 5000:
            continue


        # -------------------------------------------------
        # Ignore extremely wide red labels.
        # -------------------------------------------------

        if w > 100:
            continue


        if h > 100:
            continue


        # -------------------------------------------------
        # Question number boxes are close to square.
        # -------------------------------------------------

        ratio = w / float(h)


        if ratio < 0.5 or ratio > 2.0:
            continue


        # -------------------------------------------------
        # Main question numbers are on the left side.
        # -------------------------------------------------

        if x > width * 0.25:
            continue


        # -------------------------------------------------
        # Ignore things at the extreme top.
        #
        # This removes the large red Lesson box.
        # -------------------------------------------------

        if y < height * 0.15:
            continue


        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area
        })


    # -----------------------------------------------------
    # Sort vertically
    # -----------------------------------------------------

    candidates.sort(
        key=lambda item: item["y"]
    )


    # -----------------------------------------------------
    # Remove duplicates
    # -----------------------------------------------------

    questions = []


    for candidate in candidates:

        if not questions:

            questions.append(candidate)

            continue


        previous = questions[-1]


        if abs(
            candidate["y"] - previous["y"]
        ) < 40:

            # Keep larger box
            if candidate["area"] > previous["area"]:

                questions[-1] = candidate

        else:

            questions.append(candidate)


    return questions


# =========================================================
# CROP QUESTIONS
# =========================================================

def crop_questions(
    image_path,
    questions
):

    image = cv2.imread(image_path)

    if image is None:
        return []


    height, width = image.shape[:2]


    crops = []


    for index, question in enumerate(questions):

        # -------------------------------------------------
        # Start slightly above question number
        # -------------------------------------------------

        top = max(
            0,
            question["y"] - 20
        )


        # -------------------------------------------------
        # End before next main question
        # -------------------------------------------------

        if index < len(questions) - 1:

            next_question = questions[index + 1]

            bottom = max(
                top + 50,
                next_question["y"] - 15
            )

        else:

            bottom = height - 20


        # -------------------------------------------------
        # Safety check
        # -------------------------------------------------

        if bottom <= top:

            continue


        crop = image[
            top:bottom,
            0:width
        ]


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
# UPLOAD + ANALYZE
# =========================================================

@app.post(
    "/upload",
    response_class=HTMLResponse
)
async def upload_pdf(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".pdf"):

        return """
        <h2>❌ Please upload a PDF file.</h2>
        """


    # -----------------------------------------------------
    # Save uploaded PDF
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
    # Open PDF
    # -----------------------------------------------------

    pdf = fitz.open(
        file_path
    )


    page_count = len(pdf)


    all_questions = []


    # -----------------------------------------------------
    # Process every page
    # -----------------------------------------------------

    for page_index in range(
        page_count
    ):

        page = pdf[page_index]


        # -------------------------------------------------
        # High resolution
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
        # Detect main question boxes
        # -------------------------------------------------

        questions = detect_question_boxes(
            page_path
        )


        # -------------------------------------------------
        # Crop
        # -------------------------------------------------

        crops = crop_questions(
            page_path,
            questions
        )


        for crop in crops:

            crop["page"] = page_index + 1

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

        <title>Detected Questions</title>

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
                padding: 20px;
                background: #f0fdf4;
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
                margin-top: 15px;
                border: 1px solid #ccc;
            }}

        </style>

    </head>


    <body>

        <div class="container">

            <h1>🧠 Question Detection</h1>


            <div class="summary">

                <h2>
                    ✅ {len(all_questions)} questions detected
                </h2>

                <p>
                    PDF pages:
                    <strong>{page_count}</strong>
                </p>

            </div>
    """


    if not all_questions:

        html += """

            <div class="question">

                <h2>
                    ⚠️ No questions detected
                </h2>

                <p>
                    The detection algorithm did not find
                    the expected question markers.
                </p>

            </div>

        """


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


    if not os.path.exists(path):

        return {
            "error": "Image not found."
        }


    return FileResponse(
        path,
        media_type="image/png"
    )
