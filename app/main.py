from fastapi import FastAPI

app = FastAPI(title="Medical Doc Pipeline API", version="0.1.0")

@app.get("/health")
def health():
    print("Health check endpoint calleds2")
    return {"status": "ok"}
