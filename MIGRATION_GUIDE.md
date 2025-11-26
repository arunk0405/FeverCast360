# FeverML - SQLite to Firebase Migration Guide

## Overview
This document describes the migration from SQLite database to Firebase Firestore for the FeverML project. The migration maintains all existing functionality while leveraging Firebase's cloud-based, real-time database capabilities.

## Changes Made

### 1. Database Structure Migration

#### SQLite Tables → Firestore Collections

**Previous (SQLite)**:
- `predictions` table
- `region_metadata` table  
- `pharma_stock` table

**Current (Firebase)**:
- `predictions` collection
- `region_metadata` collection
- `pharma_stock` collection

### 2. File Changes

#### `db_utils.py` - Complete Rewrite
- **Removed**: All SQLite imports and connection logic
- **Added**: Firebase Admin SDK initialization
- **Updated Functions**:
  - `init_db()`: Now initializes Firebase connection instead of creating SQLite tables
  - `save_predictions()`: Saves to Firestore collection instead of SQLite table
  - `fetch_all_predictions()`: Fetches from Firestore with manual joins
  - `fetch_city_prediction()`: Fetches specific city data from Firestore
  - `upsert_region_metadata()`: Uses Firestore merge operations
  - `upsert_pharma_stock()`: Uses Firestore merge operations

#### `app.py` - Minor Updates
- Updated comment from "save to sqlite" to "save to Firestore"
- No functional changes - all function signatures remain the same
- UI and business logic remain unchanged

#### `prediction.py` - No Changes
- ML pipeline remains unchanged
- Still outputs the same CSV format
- No database interaction in this file

### 3. Data Model Mapping

#### Predictions Collection
```
Document ID: {region_name}
Fields:
  - region: string
  - p_outbreak: float
  - fever_type: string
  - p_type: float
  - severity_index: float
  - ts: string (timestamp)
```

#### Region Metadata Collection
```
Document ID: {region_name}
Fields:
  - lat: float
  - lon: float
  - population: integer
  - state: string
```

#### Pharma Stock Collection
```
Document ID: {region_name}
Fields:
  - paracetamol: integer
  - ors: integer
  - antibiotics: integer
  - iv_fluids: integer
  - last_updated: string (timestamp)
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Firebase Configuration

Ensure you have the Firebase Admin SDK service account key file:
- File: `newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json`
- This file should be in the project root directory
- Keep this file secure and never commit it to version control

### 3. Firestore Setup

1. Go to Firebase Console (https://console.firebase.google.com)
2. Select your project: `newp-9a65c`
3. Navigate to Firestore Database
4. If not already created, create a Firestore database
5. Set security rules as needed

**Recommended Security Rules for Development:**
```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /{document=**} {
      allow read, write: if request.auth != null;
    }
  }
}
```

**For Production, use more restrictive rules.**

### 4. Run the Application

```bash
streamlit run app.py
```

## Key Advantages of Firebase Migration

1. **Cloud-Based**: No local database file to manage
2. **Real-Time**: Changes sync in real-time across clients
3. **Scalable**: Automatic scaling without configuration
4. **Reliable**: Built-in redundancy and backup
5. **Accessible**: Can be accessed from anywhere with proper authentication
6. **No Schema Migrations**: Firestore is schema-less, making updates easier

## API Compatibility

All function signatures in `db_utils.py` remain **100% compatible** with the previous implementation:

```python
# These functions work exactly the same way
init_db()
save_predictions(df, timestamp)
fetch_all_predictions()
fetch_city_prediction(city_name)
upsert_region_metadata(region, lat, lon, population, state)
upsert_pharma_stock(region, paracetamol, ors, antibiotics, iv_fluids, ts)
```

## Data Migration (Optional)

If you have existing SQLite data to migrate:

```python
import sqlite3
import pandas as pd
from db_utils import save_predictions, upsert_region_metadata, upsert_pharma_stock
from datetime import datetime

# Connect to old SQLite database
conn = sqlite3.connect("predictions.db")

# Migrate predictions
predictions_df = pd.read_sql("SELECT * FROM predictions", conn)
if not predictions_df.empty:
    # Rename columns to match new format
    predictions_df = predictions_df.rename(columns={
        'region': 'Region',
        'p_outbreak': 'P_Outbreak',
        'fever_type': 'Fever_Type',
        'p_type': 'P_Type',
        'severity_index': 'Severity_Index'
    })
    save_predictions(predictions_df, datetime.utcnow().isoformat())

# Migrate region metadata
metadata_df = pd.read_sql("SELECT * FROM region_metadata", conn)
for _, row in metadata_df.iterrows():
    upsert_region_metadata(
        row['region'], 
        row['lat'], 
        row['lon'], 
        row.get('population'), 
        row.get('state')
    )

# Migrate pharma stock
stock_df = pd.read_sql("SELECT * FROM pharma_stock", conn)
for _, row in stock_df.iterrows():
    upsert_pharma_stock(
        row['region'],
        row['paracetamol'],
        row['ors'],
        row['antibiotics'],
        row['iv_fluids'],
        row['last_updated']
    )

conn.close()
print("Migration complete!")
```

## Testing

1. **Test Firebase Connection**:
   ```python
   from db_utils import init_db
   db = init_db()
   print("Firebase connected successfully!")
   ```

2. **Test Data Operations**:
   - Run the ML pipeline through the UI
   - Verify data appears in Firebase Console
   - Test each dashboard tab (Government, Pharma, Public views)

3. **Verify Data Integrity**:
   - Check that all predictions are saved correctly
   - Verify region metadata is properly linked
   - Confirm pharma stock calculations are accurate

## Troubleshooting

### Issue: "Firebase initialization error"
**Solution**: Verify the service account key file path and permissions

### Issue: "Import firebase_admin could not be resolved"
**Solution**: Run `pip install firebase-admin`

### Issue: "Permission denied" errors
**Solution**: Check Firestore security rules in Firebase Console

### Issue: Data not appearing
**Solution**: 
- Check Firebase Console to verify data is being written
- Verify network connectivity
- Check for any error messages in the Streamlit console

## Performance Considerations

1. **Batch Operations**: For large datasets, consider batching writes
2. **Indexing**: Create composite indexes in Firebase Console for complex queries
3. **Caching**: Consider implementing local caching for frequently accessed data
4. **Rate Limits**: Be aware of Firestore quota limits for read/write operations

## Security Best Practices

1. ✅ Keep service account key file secure
2. ✅ Never commit credentials to version control
3. ✅ Use environment variables for sensitive configuration
4. ✅ Implement proper Firestore security rules
5. ✅ Enable Firebase audit logging
6. ✅ Regularly rotate service account keys

## Rollback Plan

If you need to revert to SQLite:

1. Checkout the previous commit with SQLite code
2. Export data from Firestore using the migration script in reverse
3. Restore the `predictions.db` file
4. Update requirements.txt to remove firebase-admin

## Future Enhancements

1. **Real-time Updates**: Implement Firestore listeners for live dashboard updates
2. **User Authentication**: Add Firebase Authentication for secure access
3. **Cloud Functions**: Automate ML pipeline triggers using Cloud Functions
4. **Analytics**: Integrate Firebase Analytics for usage tracking
5. **Offline Support**: Implement Firestore offline persistence

## Support

For issues or questions:
- Check Firebase Documentation: https://firebase.google.com/docs/firestore
- Review Firestore Python SDK: https://firebase.google.com/docs/firestore/quickstart
- Check application logs in Streamlit console

---

**Migration Date**: November 25, 2025  
**Version**: 2.0.0  
**Status**: Complete ✅
