# db_utils.py
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from typing import Optional, Dict, Any

# Initialize Firebase
def init_db():
    """Initialize Firebase connection"""
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate("newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json")
            firebase_admin.initialize_app(cred)
        except Exception as e:
            print(f"Firebase initialization error: {e}")
            print("\nPlease ensure:")
            print("1. The service account key file exists")
            print("2. Cloud Firestore API is enabled at:")
            print("   https://console.developers.google.com/apis/api/firestore.googleapis.com/overview?project=newp-9a65c")
            raise
    
    try:
        return firestore.client()
    except Exception as e:
        print(f"\n⚠️  Firestore API Error: {e}")
        print("\n📝 Action Required:")
        print("   1. Enable Cloud Firestore API at:")
        print("      https://console.developers.google.com/apis/api/firestore.googleapis.com/overview?project=newp-9a65c")
        print("   2. Wait 2-3 minutes for the API to activate")
        print("   3. Restart the application")
        raise

def upsert_region_metadata(region: str, lat: Optional[float], lon: Optional[float],
                           population: Optional[int] = None, state: Optional[str] = None):
    """Update or insert region metadata in Firestore"""
    db = init_db()
    doc_ref = db.collection("region_metadata").document(region)
    
    data = {}
    if lat is not None:
        data["lat"] = lat
    if lon is not None:
        data["lon"] = lon
    if population is not None:
        data["population"] = population
    if state is not None:
        data["state"] = state
    
    # Merge to preserve existing fields
    doc_ref.set(data, merge=True)

def get_region_metadata(region: str) -> Optional[Dict[str, Any]]:
    """
    Fetches metadata for a specific region from Firestore.
    Returns dict with lat, lon, population, state if found, None otherwise.
    """
    try:
        db = init_db()
        doc_ref = db.collection("region_metadata").document(region)
        doc = doc_ref.get()
        
        if doc.exists:
            return doc.to_dict()
        return None
    except Exception as e:
        print(f"Error fetching region metadata for {region}: {e}")
        return None

def upsert_pharma_stock(region: str, paracetamol: int, ors: int, antibiotics: int, iv_fluids: int, ts: str):
    """Update or insert pharma stock data in Firestore"""
    db = init_db()
    doc_ref = db.collection("pharma_stock").document(region)
    
    data = {
        "paracetamol": paracetamol,
        "ors": ors,
        "antibiotics": antibiotics,
        "iv_fluids": iv_fluids,
        "last_updated": ts
    }
    
    doc_ref.set(data, merge=True)

def save_predictions(df: pd.DataFrame, ts: str):
    """
    Save predictions from a DataFrame to Firestore.
    DataFrame columns: Region, P_Outbreak, Fever_Type, P_Type, Severity_Index
    """
    db = init_db()
    
    for _, r in df.iterrows():
        region = str(r["Region"])
        doc_ref = db.collection("predictions").document(region)
        
        data = {
            "region": region,
            "p_outbreak": float(r["P_Outbreak"]),
            "fever_type": str(r["Fever_Type"]),
            "p_type": float(r["P_Type"]),
            "severity_index": float(r["Severity_Index"]),
            "ts": ts
        }
        
        doc_ref.set(data, merge=True)

def fetch_all_predictions() -> pd.DataFrame:
    """Fetch all predictions with joined metadata from Firestore"""
    try:
        db = init_db()
        
        # Get all predictions
        predictions_ref = db.collection("predictions").stream()
        predictions_data = []
        
        for doc in predictions_ref:
            pred = doc.to_dict()
            region = doc.id
            
            # Get region metadata
            metadata_doc = db.collection("region_metadata").document(region).get()
            metadata = metadata_doc.to_dict() if metadata_doc.exists else {}
            
            # Get pharma stock
            stock_doc = db.collection("pharma_stock").document(region).get()
            stock = stock_doc.to_dict() if stock_doc.exists else {}
            
            # Combine all data
            row = {
                "region": pred.get("region", region),
                "p_outbreak": pred.get("p_outbreak", 0),
                "fever_type": pred.get("fever_type", ""),
                "p_type": pred.get("p_type", 0),
                "severity_index": pred.get("severity_index", 0),
                "lat": metadata.get("lat"),
                "lon": metadata.get("lon"),
                "population": metadata.get("population"),
                "state": metadata.get("state"),
                "paracetamol": stock.get("paracetamol"),
                "ors": stock.get("ors"),
                "antibiotics": stock.get("antibiotics"),
                "iv_fluids": stock.get("iv_fluids"),
                "ts": pred.get("ts", "")
            }
            predictions_data.append(row)
        
        return pd.DataFrame(predictions_data) if predictions_data else pd.DataFrame()
    
    except Exception as e:
        print(f"Error fetching predictions: {e}")
        # Return empty DataFrame instead of crashing
        return pd.DataFrame()

def fetch_city_prediction(city: str) -> Optional[Dict[str, Any]]:
    """Fetch prediction data for a specific city from Firestore"""
    try:
        db = init_db()
        
        # Get prediction
        pred_doc = db.collection("predictions").document(city).get()
        if not pred_doc.exists:
            return None
        
        pred = pred_doc.to_dict()
        
        # Get region metadata
        metadata_doc = db.collection("region_metadata").document(city).get()
        metadata = metadata_doc.to_dict() if metadata_doc.exists else {}
        
        # Get pharma stock
        stock_doc = db.collection("pharma_stock").document(city).get()
        stock = stock_doc.to_dict() if stock_doc.exists else {}
        
        # Combine all data
        result = {
            "region": pred.get("region", city),
            "p_outbreak": pred.get("p_outbreak", 0),
            "fever_type": pred.get("fever_type", ""),
            "p_type": pred.get("p_type", 0),
            "severity_index": pred.get("severity_index", 0),
            "lat": metadata.get("lat"),
            "lon": metadata.get("lon"),
            "population": metadata.get("population"),
            "state": metadata.get("state"),
            "paracetamol": stock.get("paracetamol"),
            "ors": stock.get("ors"),
            "antibiotics": stock.get("antibiotics"),
            "iv_fluids": stock.get("iv_fluids"),
            "ts": pred.get("ts", "")
        }
        
        return result
    
    except Exception as e:
        print(f"Error fetching city prediction for {city}: {e}")
        return None




