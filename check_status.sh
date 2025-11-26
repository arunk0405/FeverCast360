#!/bin/bash

# FeverCast360 Project Status Check
# Verifies all components are properly configured

echo "🔍 FeverCast360 Project Status Check"
echo "===================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo "1️⃣  Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} $PYTHON_VERSION installed"
else
    echo -e "${RED}✗${NC} Python 3 not found"
    exit 1
fi

# Check virtual environment
echo ""
echo "2️⃣  Checking virtual environment..."
if [ -d "venv" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment exists"
    if [ -f "venv/requirements.installed" ]; then
        echo -e "${GREEN}✓${NC} Dependencies installed"
    else
        echo -e "${YELLOW}⚠${NC} Dependencies may need installation"
    fi
else
    echo -e "${YELLOW}⚠${NC} Virtual environment not created (run ./start.sh)"
fi

# Check Firebase credentials
echo ""
echo "3️⃣  Checking Firebase configuration..."
if [ -f "newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json" ]; then
    echo -e "${GREEN}✓${NC} Firebase credentials found"
else
    echo -e "${RED}✗${NC} Firebase credentials missing"
    echo "   Required: newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json"
fi

# Check required directories
echo ""
echo "4️⃣  Checking project structure..."
DIRS=("models" "outputs" "outputs/plots")
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} $dir/ exists"
    else
        echo -e "${YELLOW}⚠${NC} $dir/ missing (will be created automatically)"
    fi
done

# Check core files
echo ""
echo "5️⃣  Checking core files..."
FILES=("app.py" "db_utils.py" "prediction.py" "requirements.txt")
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        LINES=$(wc -l < "$file")
        echo -e "${GREEN}✓${NC} $file ($LINES lines)"
    else
        echo -e "${RED}✗${NC} $file missing"
    fi
done

# Check sample data
echo ""
echo "6️⃣  Checking sample dataset..."
if [ -f "fevercast360_sample_dataset.csv" ]; then
    ROWS=$(wc -l < "fevercast360_sample_dataset.csv")
    echo -e "${GREEN}✓${NC} Sample dataset found ($ROWS rows)"
else
    echo -e "${YELLOW}⚠${NC} Sample dataset missing"
fi

# Check documentation
echo ""
echo "7️⃣  Checking documentation..."
DOCS=("README.md" "FIREBASE_SETUP.md" "MIGRATION_GUIDE.md" "CHANGELOG.md")
DOC_COUNT=0
for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        ((DOC_COUNT++))
    fi
done
echo -e "${GREEN}✓${NC} $DOC_COUNT/$((${#DOCS[@]})) documentation files present"

# Final summary
echo ""
echo "===================================="
echo "📊 Status Summary"
echo "===================================="

# Count checks
TOTAL_CHECKS=7
PASSED_CHECKS=0

[ -f "app.py" ] && ((PASSED_CHECKS++))
[ -f "db_utils.py" ] && ((PASSED_CHECKS++))
[ -f "prediction.py" ] && ((PASSED_CHECKS++))
[ -f "requirements.txt" ] && ((PASSED_CHECKS++))
[ -f "fevercast360_sample_dataset.csv" ] && ((PASSED_CHECKS++))
command -v python3 &> /dev/null && ((PASSED_CHECKS++))
[ -f "newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json" ] && ((PASSED_CHECKS++))

if [ $PASSED_CHECKS -eq $TOTAL_CHECKS ]; then
    echo -e "${GREEN}✅ All checks passed! Project is ready.${NC}"
    echo ""
    echo "Run: ./start.sh to launch the application"
elif [ $PASSED_CHECKS -ge 5 ]; then
    echo -e "${YELLOW}⚠️  Mostly ready. Fix warnings above.${NC}"
    echo ""
    echo "You can still run: ./start.sh"
else
    echo -e "${RED}❌ Setup incomplete. Please review errors above.${NC}"
    exit 1
fi

echo ""
