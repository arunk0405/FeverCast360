# FeverCast360 - Quick Setup Guide

## 🚀 Quick Start on New Device

### 1. Copy These Files
```
✓ app.py
✓ db_utils.py  
✓ prediction.py
✓ requirements.txt
✓ newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json (IMPORTANT!)
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. Access in Browser
- Local: `http://localhost:8501`
- Network: `http://YOUR_IP:8501`

---

## 🌐 Access from Another Device (Same Network)

**On the host device:**
```bash
streamlit run app.py --server.address 0.0.0.0
```

**On another device:**
1. Find host IP address (run `ip addr` or `ifconfig` on host)
2. Open browser: `http://HOST_IP:8501`
3. Example: `http://192.168.1.100:8501`

---

## 📱 Same Firebase, Multiple Devices

Your Firebase database is **shared across all devices** automatically! 

✅ Any device with the credentials file can access the same data  
✅ All changes sync in real-time  
✅ Multiple users can view simultaneously

**Important:** Keep the `newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json` file secure and in the same folder as `app.py`

---

## 🔧 Troubleshooting

**Can't connect to Firebase?**
- Verify the `.json` file is in the correct location
- Check internet connection

**Module not found error?**
```bash
pip install [module-name]
```

**Port already in use?**
```bash
streamlit run app.py --server.port 8502
```

---

For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
