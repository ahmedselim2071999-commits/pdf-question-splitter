from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import fitz
import os
import re
import cv2
import pytesseract

from pytesseract import Output


app = FastAPI(title="PDF Question Splitter")


UPLOAD_DIR = "uploads"
RENDER_DIR = "rendered"
CROP_DIR = "crops"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RENDER_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

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
                max-width: 1000px;
                margin: auto;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 5px 25px rgba(0,0,0,0.1);
            }

            h1 {
                margin-bottom: 10px;
            }

            .upload {
                margin: 30px;
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

            .question {
                margin-top: 40px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 10px;
                background: #fafafa;
            }

            .question img {
                max-width: 100%;
                border: 1px solid #ccc;
                margin-top: 15px;
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
                class="upload"
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


# ---------------------------------------------------------
# OCR QUESTION DETECTION
# ---------------------------------------------------------

def detect_questions(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return []


    # Convert image to RGB for Tesseract
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # OCR
    data = pytesseract.image_to_data(
        rgb,
        output_type=Output.DICT,
        config="--psm 6"
    )


    candidates = []


    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        if not text:
            continue


        try:
            confidence = float(data["conf"][i])
        except:
            confidence = 0


        if confidence < 20:
            continue


        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])


        # Only simple integer numbers.
        #
        # This intentionally avoids:
        # (1)
        # (2)
        # 1.2
        # 2026
        #
        if not re.fullmatch(r"\d{1,2}", text):
            continue


        number = int(text)


        # Main question numbers are normally
        # located near the left side of the content.
        if x > image.shape[1] * 0.25:
            continue


        # Ignore tiny numbers
        if h < 10:
            continue


        candidates.append({
            "number": number,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "confidence": confidence
        })


    # -----------------------------------------------------
    # VISUAL CHECK
    #
    # Main question numbers in the example are inside
    # red boxes. We use the amount of red around the
    # detected number as an additional confidence signal.
    # -----------------------------------------------------

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    # Red color ranges
    mask1 = cv2.inRange(
        hsv,
        (0, 70, 70),
        (15, 255, 255)
    )

    mask2 = cv2.inRange(
        hsv,
        (165, 70, 70),
        (180, 255, 255)
    )

    red_mask = mask1 | mask2


    final_candidates = []


    for item in candidates:

        x = item["x"]
        y = item["y"]
        w = item["w"]
        h = item["h"]


        # Expand around the OCR number
        pad = max(8, int(h * 0.5))


        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)


        region = red_mask[y1:y2, x1:x2]


        if region.size == 0:
            continue


        red_ratio = cv2.countNonZero(region) / region.size


        # Red box increases confidence.
        if red_ratio > 0.08:
            item["visual_score"] = red_ratio
            final_candidates.append(item)


    # -----------------------------------------------------
    # SORT BY VERTICAL POSITION
    # -----------------------------------------------------

    final_candidates.sort(key=lambda q: q["y"])


    # -----------------------------------------------------
    # REMOVE DUPLICATES
    # -----------------------------------------------------

    questions = []


    for candidate in final_candidates:

        if not questions:
            questions.append(candidate)
            continue


        previous = questions[-1]


        # Same question detected more than once
        if abs(candidate["y"] - previous["y"]) < 30:

            # Keep the stronger candidate
            if candidate["confidence"] > previous["confidence"]:
                questions[-1] = candidate

            continue


        questions.append(candidate)


    return questions


# ---------------------------------------------------------
# CROP QUESTIONS
# ---------------------------------------------------------

def crop_questions(image_path, questions):

    image = cv2.imread(image_path)

    if image is None:
        return []


    height, width = image.shape[:2]


    crops = []


    for index, question in enumerate(questions):

        top = max(
            0,
            question["y"] - 20
        )


        if index < len(questions) - 1:

            next_question = questions[index + 1]

            bottom = next_question["y"] - 15

        else:

            bottom = height - 20


        if bottom <= top:
            continue


        crop = image[top:bottom, :]


        crop_path = os.path.join(
            CROP_DIR,
            f"question_{index + 1}.png"
        )


        cv2.imwrite(
            crop_path,
            crop
        )


        crops.append({
            "number": question["number"],
            "path": crop_path
        })


    return crops


# ---------------------------------------------------------
# UPLOAD + ANALYZE
# ---------------------------------------------------------

@app.post("/upload", response_class=HTMLResponse)
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):

        return """
        <h2>❌ Please upload a PDF file.</h2>
        """


    # Save PDF
    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )


    with open(file_path, "wb") as buffer:

        buffer.write(
            await file.read()
        )


    # Open PDF
    pdf = fitz.open(file_path)


    page_count = len(pdf)


    # -----------------------------------------------------
    # FOR V1:
    # Process every page.
    # -----------------------------------------------------

    all_questions = []


    for page_index in range(page_count):

        page = pdf[page_index]


        # High-resolution rendering
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


        pix.save(page_path)


        # Detect questions
        questions = detect_questions(
            page_path
        )


        # Crop questions
        crops = crop_questions(
            page_path,
            questions
        )


        for crop in crops:

            crop["page"] = page_index + 1

            all_questions.append(crop)


    pdf.close()


    # -----------------------------------------------------
    # BUILD RESULT PAGE
    # -----------------------------------------------------

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

            .success {{
                color: #15803d;
            }}

        </style>

    </head>

    <body>

        <div class="container">

            <h1>🧠 Question Detection</h1>

            <h2 class="success">
                ✅ {len(all_questions)} questions detected
            </h2>

            <p>
                PDF pages: {page_count}
            </p>
    """


    if not all_questions:

        html += """
            <h2>⚠️ No questions detected.</h2>

            <p>
                We will improve the detection algorithm.
            </p>
        """


    for index, question in enumerate(all_questions):

        html += f"""

            <div class="question">

                <h2>
                    Question {index + 1}
                </h2>

                <p>
                    Detected number:
                    <strong>{question["number"]}</strong>
                    |
                    Page:
                    <strong>{question["page"]}</strong>
                </p>

                <img
                    src="/crop/{os.path.basename(question["path"])}"
                >

            </div>

        """


    html += """

        </div>

    </body>

    </html>
    """


    return html


# ---------------------------------------------------------
# SERVE CROPPED IMAGES
# ---------------------------------------------------------

@app.get("/crop/{filename}")
def get_crop(filename):

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
