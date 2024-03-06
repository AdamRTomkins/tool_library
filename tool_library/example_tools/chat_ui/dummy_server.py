from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Directory where uploaded files will be stored
UPLOAD_DIRECTORY = "static"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Mount the static directory to be served by FastAPI
app.mount("/static", StaticFiles(directory="static"), name="static")

# Endpoint to search the index
@app.get("/search/", tags=["Search"])
async def search_index(query: str = ""):
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required")
    
    return {"results":["yup"]}

@app.post("/index/")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_DIRECTORY}/{file.filename.split('/')[-1]}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    return {"filename": file.filename, "url": f"/static/{file.filename.split('/')[-1]}"}

@app.get("/")
async def main():
    content = """
<body>
<form action="/uploadfile/" enctype="multipart/form-data" method="post">
<input name="file" type="file">
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
