# Changelog

All notable changes to FeverCast360 will be documented in this file.

## [2.0.0] - 2025-11-25

### 🎉 Major Release - Dynamic Data & Code Cleanup

#### Added
- **Dynamic Geocoding System**: Automatic city coordinate fetching via OpenStreetMap
- **Smart Caching**: Reuses existing geocoded data to minimize API calls
- **Multi-Strategy Geocoding**: Tries multiple search patterns for better accuracy
- **Re-Geocoding Tool**: Manual retry option for failed cities
- **Comprehensive README**: Detailed documentation with usage guides
- **Quick Start Script**: One-command setup and launch (`start.sh`)
- **.gitignore**: Proper exclusion of sensitive and temporary files
- **Professional Styling**: White background theme with dark text throughout

#### Changed
- **Removed Static Data Dependency**: Eliminated hardcoded CITY_COORDS dictionary
- **All Components Use Live Data**: Map, pharma, and government views pull from Firebase
- **Enhanced Error Handling**: Better feedback during geocoding and API calls
- **Improved Progress Indicators**: Detailed status during ML pipeline execution

#### Removed
- ❌ All commented-out legacy code (600+ lines removed from app.py)
- ❌ SQLite database files (predictions.db)
- ❌ Duplicate Firebase utility files (firebase_utils.py, googlefirbase_utils.py)
- ❌ Obsolete setup scripts (check_firebase_setup.py)
- ❌ Redundant documentation (UI_IMPROVEMENTS.md)

#### Fixed
- ✅ Text visibility on white background (comprehensive CSS updates)
- ✅ Deprecation warnings (use_container_width, datetime.utcnow)
- ✅ Map only showing hardcoded cities
- ✅ Missing cities not appearing on map

### Code Quality Improvements
- **Reduced Total Lines**: From 2,900+ to 2,068 (29% reduction)
  - app.py: 2077 → 1401 lines (32% reduction)
  - db_utils.py: 245 → 199 lines (19% reduction)
  - prediction.py: 571 → 468 lines (18% reduction)
- **Removed Dead Code**: Eliminated all commented legacy implementations
- **Better Organization**: Clear separation of concerns
- **Enhanced Documentation**: Comprehensive inline comments

---

## [1.0.0] - 2024-11-24

### Initial Release

#### Features
- Two-stage ML pipeline (Logistic Regression + Random Forest/XGBoost)
- Firebase Firestore integration
- Interactive Leaflet maps with city boundaries
- Three dashboard views (Government, Pharma, Public)
- Basic geocoding for major cities
- Professional UI with gradient cards and responsive design

---

## Version Naming Convention

- **Major (X.0.0)**: Breaking changes, major feature additions
- **Minor (0.X.0)**: New features, non-breaking changes
- **Patch (0.0.X)**: Bug fixes, minor improvements

---

## Coming Soon 🚀

### Planned for v2.1.0
- [ ] Bulk geocoding improvement with parallel processing
- [ ] Export functionality for predictions and reports
- [ ] Historical trend analysis and visualization
- [ ] Mobile-responsive design improvements
- [ ] API endpoint for external integrations

### Planned for v3.0.0
- [ ] Real-time data streaming
- [ ] Advanced ML models (LSTM, Prophet)
- [ ] Multi-language support
- [ ] Alert notification system (Email/SMS)
- [ ] Integration with weather APIs
- [ ] Automated report generation

---

**Note**: This changelog follows [Keep a Changelog](https://keepachangelog.com/) format.
