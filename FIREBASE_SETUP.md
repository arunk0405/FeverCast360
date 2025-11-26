# ⚠️ Firebase Setup Required

## Current Status

Your FeverML application has been successfully migrated from SQLite to Firebase, but the **Cloud Firestore API** needs to be enabled in your Google Cloud project.

## 🚀 Quick Fix (2 minutes)

### Step 1: Enable Firestore API

Click this link to enable the API:

**👉 https://console.developers.google.com/apis/api/firestore.googleapis.com/overview?project=newp-9a65c**

Then click the **"ENABLE"** button.

### Step 2: Wait for Activation

After enabling, wait **2-3 minutes** for the API to activate across Google's systems.

### Step 3: Create Firestore Database (First Time Only)

1. Go to Firebase Console: https://console.firebase.google.com/project/newp-9a65c
2. Click on **"Firestore Database"** in the left menu
3. Click **"Create database"**
4. Choose a location (e.g., "us-central" or closest to you)
5. Start in **"Production mode"** (you can adjust rules later)
6. Click **"Enable"**

### Step 4: Verify Setup

Run the setup checker:

```bash
cd /home/arun-k/FeverML
venv/bin/python check_firebase_setup.py
```

You should see all checks passing ✅

### Step 5: Run Your Application

```bash
cd /home/arun-k/FeverML
source venv/bin/activate
streamlit run app.py
```

## 🔧 What Changed?

- ✅ **Code**: Migrated from SQLite to Firebase (complete)
- ✅ **Dependencies**: firebase-admin installed (complete)
- ✅ **Credentials**: Service account key file present (complete)
- ⏳ **API**: Firestore API needs to be enabled (action required)

## 📋 Checklist

- [ ] Enable Firestore API (Step 1)
- [ ] Wait 2-3 minutes (Step 2)
- [ ] Create Firestore database (Step 3)
- [ ] Run setup checker (Step 4)
- [ ] Launch application (Step 5)

## 🆘 Troubleshooting

### Error: "403 Permission Denied"
- **Cause**: Firestore API not enabled yet
- **Fix**: Follow Step 1 above

### Error: "Service account key file not found"
- **Cause**: Credentials file missing
- **Fix**: Ensure `newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json` is in the project root

### Error: "Module not found: firebase_admin"
- **Cause**: Package not installed in virtual environment
- **Fix**: `source venv/bin/activate && pip install firebase-admin`

## 📚 Additional Resources

- [Firebase Console](https://console.firebase.google.com/project/newp-9a65c)
- [Firestore Documentation](https://firebase.google.com/docs/firestore)
- [Migration Guide](MIGRATION_GUIDE.md) - Detailed migration documentation

## ✨ Benefits After Setup

Once enabled, your application will have:

- 🌐 **Cloud-based database** - Access from anywhere
- 🔄 **Real-time updates** - Instant data synchronization
- 📈 **Auto-scaling** - Handles any load automatically
- 🔒 **Built-in security** - Enterprise-grade protection
- 💾 **Automatic backups** - No data loss risk

---

**Need Help?** Check the [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed information.
