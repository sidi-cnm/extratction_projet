from fastapi import APIRouter, UploadFile, File, HTTPException
from ..schemas import ExtractRequest, ExtractResponse, ExtractFileResponse
from ..services.extractor import process_text, extract_text_from_pdf

router = APIRouter(prefix="", tags=["extract"])

@router.post("/extract", response_model=ExtractResponse)
def extract_from_text(req: ExtractRequest):
    try:
        data, valid, err = process_text(req.text)
        return ExtractResponse(json=data, valid=valid, validation_error=err)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-file", response_model=ExtractFileResponse)
def extract_from_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Merci d'uploader un PDF.")
    try:
        # Sauver temporairement puis extraire
        content = file.file.read()
        tmp_path = f"/tmp/{file.filename}"
        with open(tmp_path, "wb") as f:
            f.write(content)

        text = extract_text_from_pdf(tmp_path)
        data, valid, err = process_text(text)

        return ExtractFileResponse(
            filename=file.filename,
            size=len(content),
            content_type=file.content_type or "application/pdf",
            json=data,
            valid=valid,
            validation_error=err
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
