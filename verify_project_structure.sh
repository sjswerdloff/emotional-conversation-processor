#!/bin/bash

# Verify Emotional Conversation Processor Project Structure
# This script checks if all files are in their correct locations

echo "🔍 Verifying project structure..."
echo ""

# Function to check if file exists and report
check_file() {
    local file_path="$1"
    local description="$2"

    if [ -f "$file_path" ]; then
        echo "✅ $description: $file_path"
        return 0
    else
        echo "❌ MISSING $description: $file_path"
        return 1
    fi
}

# Function to check if directory exists
check_dir() {
    local dir_path="$1"
    local description="$2"

    if [ -d "$dir_path" ]; then
        echo "✅ $description: $dir_path/"
        return 0
    else
        echo "❌ MISSING $description: $dir_path/"
        return 1
    fi
}

missing_count=0

echo "📦 CORE SOURCE FILES:"
check_file "src/emotional_processor/core/models.py" "Core models" || ((missing_count++))
check_file "src/emotional_processor/processors/emotion_classifier.py" "Emotion classifier" || ((missing_count++))
check_file "src/emotional_processor/processors/technical_detector.py" "Technical detector" || ((missing_count++))
check_file "src/emotional_processor/processors/conversation_segmenter.py" "Conversation segmenter" || ((missing_count++))
check_file "src/emotional_processor/embeddings/emotion_aware_embedder.py" "Emotion-aware embedder" || ((missing_count++))
check_file "src/emotional_processor/storage/vector_store.py" "Vector store" || ((missing_count++))

echo ""
echo "📜 SCRIPT FILES:"
check_file "scripts/process_conversation.py" "Main processing script" || ((missing_count++))
check_file "scripts/process_conversation_enhanced.py" "Enhanced processing script" || ((missing_count++))
check_file "scripts/setup_database.py" "Database setup script" || ((missing_count++))
check_file "scripts/health_check.py" "Health check script" || ((missing_count++))
check_file "scripts/clean_conversation.py" "Conversation cleaner" || ((missing_count++))
check_file "scripts/claude_conversation_extractor.js" "Claude conversation extractor" || ((missing_count++))
check_file "scripts/safari_conversation_extractor.js" "Safari conversation extractor" || ((missing_count++))

echo ""
echo "⚙️  CONFIGURATION FILES:"
check_file "config/development.yaml" "Development config" || ((missing_count++))
check_file "config/testing.yaml" "Testing config" || ((missing_count++))
check_file ".flake8" "Flake8 config" || ((missing_count++))
check_file "mypy.ini" "MyPy config" || ((missing_count++))
check_file "pytest.ini" "Pytest config" || ((missing_count++))
check_file "pyproject.toml" "Project config" || ((missing_count++))
check_file "requirements.txt" "Requirements" || ((missing_count++))
check_file "requirements-dev.txt" "Dev requirements" || ((missing_count++))
check_file "setup.py" "Setup script" || ((missing_count++))

echo ""
echo "🧪 TEST FILES:"
check_file "tests/conftest.py" "Test configuration" || ((missing_count++))
check_file "tests/unit/test_emotion_classifier.py" "Emotion classifier tests" || ((missing_count++))

echo ""
echo "🐳 DOCKER FILES:"
check_file "docker/docker-compose.yml" "Docker compose" || ((missing_count++))
check_file "docker/Dockerfile" "Dockerfile" || ((missing_count++))

echo ""
echo "📁 DIRECTORY STRUCTURE:"
check_dir "src/emotional_processor" "Main source directory"
check_dir "src/emotional_processor/core" "Core module"
check_dir "src/emotional_processor/processors" "Processors module"
check_dir "src/emotional_processor/embeddings" "Embeddings module"
check_dir "src/emotional_processor/storage" "Storage module"
check_dir "tests/unit" "Unit tests"
check_dir "tests/integration" "Integration tests"
check_dir "scripts" "Scripts directory"
check_dir "config" "Configuration directory"
check_dir "docker" "Docker directory"

echo ""
echo "🐍 PYTHON PACKAGE FILES:"
python_packages=(
    "src/emotional_processor/__init__.py"
    "src/emotional_processor/core/__init__.py"
    "src/emotional_processor/processors/__init__.py"
    "src/emotional_processor/embeddings/__init__.py"
    "src/emotional_processor/storage/__init__.py"
    "tests/__init__.py"
    "tests/unit/__init__.py"
    "scripts/__init__.py"
)

init_missing=0
for init_file in "${python_packages[@]}"; do
    if [ -f "$init_file" ]; then
        echo "✅ Package init: $init_file"
    else
        echo "❌ MISSING init: $init_file"
        ((init_missing++))
    fi
done

echo ""
echo "📊 SUMMARY:"
echo "================="
if [ $missing_count -eq 0 ] && [ $init_missing -eq 0 ]; then
    echo "🎉 ALL FILES PROPERLY ORGANIZED!"
    echo "✅ Project structure is correct"
    echo ""
    echo "🚀 Ready for development:"
    echo "  pip install -r requirements-dev.txt"
    echo "  docker-compose -f docker/docker-compose.yml up -d qdrant"
    echo "  python scripts/setup_database.py"
    echo "  pytest"
else
    echo "⚠️  Issues found:"
    echo "  Missing core files: $missing_count"
    echo "  Missing __init__.py files: $init_missing"
    echo ""
    echo "🔧 To fix:"
    echo "  1. Run the organize_files.sh script"
    echo "  2. Manually create any missing files"
    echo "  3. Check for typos in filenames"
fi

echo ""
echo "📋 Current directory contents:"
ls -la src/emotional_processor/ 2>/dev/null || echo "❌ src/emotional_processor/ directory not found"
