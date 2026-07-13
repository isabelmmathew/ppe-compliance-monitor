from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import tempfile
import shutil
import os


from pipeline import process_video

app = FastAPI(title = "PPE Compliance Detector", 
              description="The API analyses each frame of a given video using a trained YOLO model, tracks PPE detections and reports violations", 
              version = "1.0.0")

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

latest_csv = None
latest_video = None

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/process-video", 
          summary = "Process a construction site video", 
          description="Upload an mp4 video to receive an annoted video showing PPE compliance")
async def process(background_tasks: BackgroundTasks, video: UploadFile = File(...)):

    if not video.filename:
        raise HTTPException(status_code=400, detail = "No file uploaded")
    
    extension = os.path.splitext(video.filename)[1].lower()

    if extension != ".mp4":
        raise HTTPException(status_code= 414, detail = "Only mp4 files supported")

    # Create temporary input file
    temp_input = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    )

    # Copy uploaded video into temporary file
    shutil.copyfileobj(video.file, temp_input)
    temp_input.close()

    if os.path.getsize(temp_input.name) == 0:
        os.remove(temp_input.name)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    )
    temp_output.close()

    # Run PPE pipeline
    try:
        output_path, csv_path = process_video(
            temp_input.name,
            temp_output.name
        )

        global latest_video, latest_csv

        latest_video = output_path
        latest_csv = csv_path
        
    except Exception as e:
        os.remove(temp_input.name)
        os.remove(temp_output.name)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")

    # Schedule cleanup
    background_tasks.add_task(os.remove, temp_input.name)
    background_tasks.add_task(os.remove, temp_output.name)

    # Return processed video
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename="processed_video.mp4"
    )

@app.get("/download-report")
def download_report():

    if latest_video is None or latest_csv is None:
        raise HTTPException(
            status_code=404,
            detail="No report available."
        )

    zip_path = tempfile.NamedTemporaryFile(
        suffix=".zip",
        delete=False
    ).name

    import zipfile

    with zipfile.ZipFile(zip_path, "w") as zipf:

        zipf.write(
            latest_video,
            arcname="processed_video.mp4"
        )

        zipf.write(
            latest_csv,
            arcname="violations.csv"
        )

        for file in os.listdir("snapshots"):

            zipf.write(
                os.path.join("snapshots", file),
                arcname=os.path.join("snapshots", file)
            )

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="Complete_Report.zip"
    )