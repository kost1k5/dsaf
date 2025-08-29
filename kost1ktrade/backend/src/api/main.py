from fastapi import FastAPI

app = FastAPI(title="Kost1kTrade API")

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Check if the API is running."""
    return {"status": "ok"}
