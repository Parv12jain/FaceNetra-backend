# from fastapi import FastAPI, UploadFile, File
# from backend.recognition import recognize_face_from_image
# from backend.attendance import mark_attendance

# app = FastAPI(title="Vision Attendance API")


# @app.post("/recognize")
# async def recognize(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     names = recognize_face_from_image(image_bytes)

#     marked = []
#     for name in names:
#         if name != "Unknown":
#             if mark_attendance(name):
#                 marked.append(name)

#     return {
#         "detected_faces": names,
#         "attendance_marked": marked
#     }


# @app.get("/")
# def root():
#     return {"message": "Vision Attendance API is running"}


from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from recognition import recognize_face

app = FastAPI(title="FaceNetra Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        file_path = f"/tmp/{file.filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        result = recognize_face(file_path)

        return {"result": result}

    except Exception as e:
        return {"error": str(e)}

