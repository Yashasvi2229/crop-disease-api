from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict, chat, market_prices

app = FastAPI(title="AgroWise Crop Disease Detector API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/api", tags=["predict"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(market_prices.router, prefix="/api", tags=["market_prices"])

@app.get("/")
def read_root():
    return {
        "message": "AgroWise Crop Disease Detector API with AI Chat, Weather & Market Prices",
        "endpoints": ["/api/predict", "/api/chat", "/api/market-prices"]
    }
