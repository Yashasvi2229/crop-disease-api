from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.predict import router as predict_router

app = FastAPI(title="Crop Disease Classification API")

# Add CORS middleware to allow web browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the predict route
app.include_router(predict_router)

@app.get("/")
def home():
    return {"message": "Welcome to the Crop Disease Classification API"}
