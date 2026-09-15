from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import fitz
import os

app = FastAPI(title="PDF Question Splitter")


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
                padding: 60px;
            }

            .box {
                background: white;
                max-width: 700px;
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
                Upload a PDF containing questions
                and split them into PowerPoint slides.
            </p>

            <form action="/upload" method="post" enctype="multipart/form-data">

                <input
                    type="file"
                    name="file"
                    accept=".pdf"
                    required
                >

                <br>

                <button type="submit">
                    Upload PDF
                </button>

            </form>

        </div>

    </body>
    </html>
    """


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Please upload a PDF file."}

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join("uploads", file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Open PDF
    pdf = fitz.open(file_path)

    pages = len(pdf)

    pdf.close()

    return {
        "filename": file.filename,
        "pages": pages,
        "message": "PDF uploaded successfully."
    }
