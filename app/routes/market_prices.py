"""
Market prices routes for commodity price data
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import requests
from datetime import datetime

router = APIRouter()

# data.gov.in API endpoint for commodity prices
DATA_GOV_API = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

@router.get("/market-prices")
async def get_market_prices(
    state: Optional[str] = None,
    commodity: Optional[str] = None,
    limit: int = 100
):
    """
    Get market prices from data.gov.in
    
    Args:
        state: State name filter (optional)
        commodity: Commodity name filter (optional)
        limit: Maximum number of results (default 100)
    
    Returns:
        List of commodity prices with trends
    """
    try:
        # Build API parameters
        params = {
            "api-key": "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b",
            "format": "json",
            "limit": limit,
            "offset": 0
        }
        
        # Add filters if provided
        filters = {}
        if state:
            filters["state"] = state
        if commodity:
            filters["commodity"] = commodity
            
        if filters:
            params["filters"] = filters
        
        # Fetch data from data.gov.in
        response = requests.get(DATA_GOV_API, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if "records" not in data:
            return {
                "success": False,
                "message": "No data available",
                "crops": []
            }
        
        # Process and format the data
        crops = []
        seen_commodities = set()
        
        for record in data["records"]:
            commodity_name = record.get("commodity", "Unknown")
            
            # Skip duplicates (keep only latest price for each commodity)
            if commodity_name in seen_commodities:
                continue
            seen_commodities.add(commodity_name)
            
            # Extract price data
            modal_price = record.get("modal_price", 0)
            min_price = record.get("min_price", 0)
            max_price = record.get("max_price", 0)
            
            # Calculate average price
            try:
                price = float(modal_price) if modal_price else float(max_price)
            except (ValueError, TypeError):
                price = 0
            
            if price == 0:
                continue
            
            # Determine trend (simplified - would need historical data for real trend)
            # For now, use price variance as indicator
            try:
                price_variance = (float(max_price) - float(min_price)) / float(modal_price) * 100 if modal_price else 0
                if price_variance > 5:
                    trend = "up"
                    change = abs(price_variance)
                elif price_variance < -5:
                    trend = "down"
                    change = abs(price_variance)
                else:
                    trend = "stable"
                    change = 0
            except (ValueError, TypeError, ZeroDivisionError):
                trend = "stable"
                change = 0
            
            crop_data = {
                "name": commodity_name,
                "price": round(price, 2),
                "unit": "quintal",  # Most commodities are per quintal
                "trend": trend,
                "change": round(change, 1),
                "market": record.get("market", ""),
                "state": record.get("state", ""),
                "district": record.get("district", ""),
                "date": record.get("arrival_date", "")
            }
            
            crops.append(crop_data)
        
        return {
            "success": True,
            "count": len(crops),
            "crops": crops
        }
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch market data: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/market-prices/trending")
async def get_trending_crops(state: Optional[str] = None, limit: int = 10):
    """
    Get trending crops with highest price changes
    
    Args:
        state: State filter (optional)
        limit: Number of crops to return (default 10)
    
    Returns:
        List of trending crops
    """
    # Get all market prices
    result = await get_market_prices(state=state, limit=200)
    
    if not result["success"]:
        return result
    
    # Sort by absolute change
    crops = sorted(
        result["crops"],
        key=lambda x: abs(x["change"]),
        reverse=True
    )[:limit]
    
    return {
        "success": True,
        "count": len(crops),
        "crops": crops
    }
