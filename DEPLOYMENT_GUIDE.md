# FeverCast360 Deployment Guide

## Deploy on Another Device (Same Firebase Database)

### Prerequisites
- Python 3.8 or higher
- Internet connection
- The Firebase service account key file

---

## Method 1: Local Network Access (Easiest)

If both devices are on the same network, you can access the app directly:

1. **On the current device**, run:
   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```

2. **Find your IP address**:
   - Linux/Mac: `ip addr show` or `ifconfig`
   - Windows: `ipconfig`
   
3. **Access from another device**:
   - Open browser and go to: `http://YOUR_IP_ADDRESS:8501`
   - Example: `http://192.168.1.100:8501`

---

## Method 2: Complete Installation on New Device

### Step 1: Transfer Project Files

Copy these files to the new device:

**Required Files:**
```
app.py
db_utils.py
prediction.py
newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json  ← CRITICAL!
fevercast360_sample_dataset.csv (optional, for testing)
```

**Transfer Methods:**
- USB drive
- Git repository (recommended - but add `.json` to `.gitignore` for security)
- Cloud storage (Google Drive, Dropbox)
- Email (not recommended for credentials)
- SCP/SFTP for remote servers

### Step 2: Install Python Dependencies

Create a `requirements.txt` file (if not present):
```txt
streamlit==1.28.0
firebase-admin==6.2.0
pandas==2.1.0
numpy==1.24.3
folium==0.14.0
streamlit-folium==0.15.0
plotly==5.17.0
scikit-learn==1.3.0
joblib==1.3.2
requests==2.31.0
geopy==2.4.0
xgboost==2.0.0  # Optional, for advanced ML
```

Install dependencies:
```bash
# Using pip
pip install -r requirements.txt

# Or install individually
pip install streamlit firebase-admin pandas numpy folium streamlit-folium plotly scikit-learn joblib requests geopy
```

### Step 3: Verify Firebase Credentials

Make sure the Firebase JSON file is in the same directory as `app.py`:

```
/your/project/folder/
├── app.py
├── db_utils.py
├── prediction.py
└── newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json  ← Must be here
```

The `db_utils.py` file should have this line (already configured):
```python
cred = credentials.Certificate("newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json")
```

### Step 4: Run the Application

```bash
cd /path/to/your/project
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Method 3: Deploy to Cloud (Internet Access from Anywhere)

### Option A: Streamlit Community Cloud (FREE)

1. **Push code to GitHub**:
   ```bash
   git init
   git add app.py db_utils.py prediction.py requirements.txt
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/fevercast360.git
   git push -u origin main
   ```

2. **⚠️ IMPORTANT**: Do NOT commit the Firebase JSON file to GitHub!
   
   Create `.gitignore`:
   ```
   newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json
   __pycache__/
   *.pyc
   .env
   ```

3. **Deploy on Streamlit Cloud**:
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Add Firebase credentials via Streamlit Secrets:
     - Go to App Settings → Secrets
     - Add your Firebase JSON content as a secret

4. **Update `db_utils.py` to use secrets**:
   ```python
   import streamlit as st
   import json
   
   # Check if running on Streamlit Cloud
   if "firebase_credentials" in st.secrets:
       cred_dict = json.loads(st.secrets["firebase_credentials"])
       cred = credentials.Certificate(cred_dict)
   else:
       # Local development
       cred = credentials.Certificate("newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json")
   ```

### Option B: Heroku, AWS, or Google Cloud

Similar process but requires more configuration. Contact me if you need help with these platforms.

---

## Security Best Practices

### 🔒 Protecting Your Firebase Credentials

1. **Never commit credentials to Git**:
   ```bash
   echo "*.json" >> .gitignore
   git rm --cached newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json
   ```

2. **Use environment variables** (optional):
   ```python
   import os
   import json
   
   # Load from environment
   firebase_creds = os.getenv('FIREBASE_CREDENTIALS')
   if firebase_creds:
       cred_dict = json.loads(firebase_creds)
       cred = credentials.Certificate(cred_dict)
   ```

3. **Restrict Firebase permissions**:
   - Go to Firebase Console → Project Settings
   - Service Accounts → Manage permissions
   - Limit access to only what's needed

---

## Troubleshooting

### Issue: "Firebase app already initialized"
**Solution**: Restart the Streamlit app or clear the cache

### Issue: "Permission denied" or authentication errors
**Solution**: 
- Verify the Firebase JSON file is present
- Check file path is correct
- Ensure Firebase project is active in the console

### Issue: "Module not found"
**Solution**: Install missing dependencies:
```bash
pip install [missing-module-name]
```

### Issue: Cannot access from another device on network
**Solution**: 
1. Run with `--server.address 0.0.0.0`
2. Check firewall settings
3. Ensure both devices are on the same network

### Issue: Data not syncing
**Solution**: 
- Check internet connection
- Verify Firebase Firestore rules allow read/write
- Check Firebase Console for any quota limits

---

## Quick Start Checklist

- [ ] Copy all Python files to new device
- [ ] Copy Firebase JSON credentials file
- [ ] Install Python 3.8+
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify Firebase JSON is in the same directory
- [ ] Run: `streamlit run app.py`
- [ ] Access at `http://localhost:8501`

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all files are present
3. Check Firebase Console for any errors
4. Review Streamlit logs in terminal

**Firebase Console**: https://console.firebase.google.com/
**Firestore Database**: Check the "Firestore Database" section in your project

---

## File Structure

```
FeverML/
├── app.py                                          # Main Streamlit application
├── db_utils.py                                     # Firebase utilities
├── prediction.py                                   # ML pipeline
├── newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json  # Firebase credentials
├── requirements.txt                                 # Python dependencies
├── fevercast360_sample_dataset.csv                 # Sample data (optional)
├── models/                                         # ML models (auto-generated)
├── outputs/                                        # Predictions (auto-generated)
└── __pycache__/                                    # Python cache (auto-generated)
```

---

**Last Updated**: November 26, 2025
