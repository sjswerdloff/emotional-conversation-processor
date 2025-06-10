#!/bin/bash

# Organize Emotional Conversation Processor Files
# This script moves files from their current scattered locations to the proper directory structure

echo "🔧 Organizing Emotional Conversation Processor files..."

# Create any missing directories first
echo "📁 Creating missing directories..."
mkdir -p src/emotional_processor/{core,processors,embeddings,storage,retrieval,utils,api/endpoints,api/middleware,config}
mkdir -p tests/{unit,integration,performance,e2e,fixtures/test_conversations}
mkdir -p scripts
mkdir -p config
mkdir -p docs/{examples,architecture}
mkdir -p docker/qdrant
mkdir -p monitoring/{prometheus,grafana/dashboards,alerts}
mkdir -p notebooks
mkdir -p data/{raw/conversations,processed/embeddings,models/custom_models,exports/replayed_conversations}
mkdir -p logs/{application,tests}

# Move core source files
echo "📦 Moving core source files..."
if [ -f "core_models.py" ]; then
    mv core_models.py src/emotional_processor/core/models.py
    echo "  ✅ Moved core_models.py -> src/emotional_processor/core/models.py"
fi

if [ -f "emotion_classifier.py" ]; then
    mv emotion_classifier.py src/emotional_processor/processors/emotion_classifier.py
    echo "  ✅ Moved emotion_classifier.py -> src/emotional_processor/processors/emotion_classifier.py"
fi

if [ -f "technical_detector.py" ]; then
    mv technical_detector.py src/emotional_processor/processors/technical_detector.py
    echo "  ✅ Moved technical_detector.py -> src/emotional_processor/processors/technical_detector.py"
fi

if [ -f "enhanced_conversation_segmenter.py" ]; then
    mv enhanced_conversation_segmenter.py src/emotional_processor/processors/conversation_segmenter.py
    echo "  ✅ Moved enhanced_conversation_segmenter.py -> src/emotional_processor/processors/conversation_segmenter.py"
fi

if [ -f "vector_store.py" ]; then
    mv vector_store.py src/emotional_processor/storage/vector_store.py
    echo "  ✅ Moved vector_store.py -> src/emotional_processor/storage/vector_store.py"
fi

# Move embeddings file from embeddings directory
if [ -f "embeddings/emotion_aware_embedder.py" ]; then
    mv embeddings/emotion_aware_embedder.py src/emotional_processor/embeddings/emotion_aware_embedder.py
    echo "  ✅ Moved embeddings/emotion_aware_embedder.py -> src/emotional_processor/embeddings/emotion_aware_embedder.py"
    # Remove the now-empty embeddings directory
    rmdir embeddings 2>/dev/null
fi

# Move script files
echo "📜 Moving script files..."
if [ -f "main_processing_script.py" ]; then
    mv main_processing_script.py scripts/process_conversation.py
    echo "  ✅ Moved main_processing_script.py -> scripts/process_conversation.py"
fi

if [ -f "enhanced_process_conversation.py" ]; then
    mv enhanced_process_conversation.py scripts/process_conversation_enhanced.py
    echo "  ✅ Moved enhanced_process_conversation.py -> scripts/process_conversation_enhanced.py"
fi

if [ -f "setup_database_script.py" ]; then
    mv setup_database_script.py scripts/setup_database.py
    echo "  ✅ Moved setup_database_script.py -> scripts/setup_database.py"
fi

if [ -f "health_check_script.py" ]; then
    mv health_check_script.py scripts/health_check.py
    echo "  ✅ Moved health_check_script.py -> scripts/health_check.py"
fi

if [ -f "claude_conversation_extractor.js" ]; then
    mv claude_conversation_extractor.js scripts/claude_conversation_extractor.js
    echo "  ✅ Moved claude_conversation_extractor.js -> scripts/claude_conversation_extractor.js"
fi

if [ -f "safari_json_conversation_extractor.js" ]; then
    mv safari_json_conversation_extractor.js scripts/safari_conversation_extractor.js
    echo "  ✅ Moved safari_json_conversation_extractor.js -> scripts/safari_conversation_extractor.js"
fi

# Move any additional scripts that might exist
if [ -f "json_to_text_converter.py" ]; then
    mv json_to_text_converter.py scripts/json_to_text_converter.py
    echo "  ✅ Moved json_to_text_converter.py -> scripts/json_to_text_converter.py"
fi

if [ -f "json_conversation_extractor.js" ]; then
    mv json_conversation_extractor.js scripts/json_conversation_extractor.js
    echo "  ✅ Moved json_conversation_extractor.js -> scripts/json_conversation_extractor.js"
fi

# Move configuration files
echo "⚙️  Moving configuration files..."
if [ -f "config_development.yaml" ]; then
    mv config_development.yaml config/development.yaml
    echo "  ✅ Moved config_development.yaml -> config/development.yaml"
fi

if [ -f "config_testing.yaml" ]; then
    mv config_testing.yaml config/testing.yaml
    echo "  ✅ Moved config_testing.yaml -> config/testing.yaml"
fi

if [ -f "flake8.config" ]; then
    mv flake8.config .flake8
    echo "  ✅ Moved flake8.config -> .flake8"
fi

# Move test files
echo "🧪 Moving test files..."
if [ -f "test_conftest.py" ]; then
    mv test_conftest.py tests/conftest.py
    echo "  ✅ Moved test_conftest.py -> tests/conftest.py"
fi

if [ -f "test_emotional_classifier.py" ] && [ -d "tests/unit" ]; then
    # If it's not already in the right place
    if [ ! -f "tests/unit/test_emotion_classifier.py" ]; then
        mv test_emotional_classifier.py tests/unit/test_emotion_classifier.py
        echo "  ✅ Moved test_emotional_classifier.py -> tests/unit/test_emotion_classifier.py"
    fi
fi

# Move docker files (if they exist in root)
echo "🐳 Moving Docker files..."
if [ -f "docker-compose.yml" ] && [ ! -f "docker/docker-compose.yml" ]; then
    mv docker-compose.yml docker/docker-compose.yml
    echo "  ✅ Moved docker-compose.yml -> docker/docker-compose.yml"
fi

if [ -f "Dockerfile" ] && [ ! -f "docker/Dockerfile" ]; then
    mv Dockerfile docker/Dockerfile
    echo "  ✅ Moved Dockerfile -> docker/Dockerfile"
fi

# Make scripts executable
echo "🔐 Making scripts executable..."
chmod +x scripts/*.py 2>/dev/null
chmod +x scripts/*.sh 2>/dev/null
echo "  ✅ Made scripts executable"

# Create __init__.py files for Python packages
echo "🐍 Creating __init__.py files..."
touch src/emotional_processor/__init__.py
touch src/emotional_processor/core/__init__.py
touch src/emotional_processor/processors/__init__.py
touch src/emotional_processor/embeddings/__init__.py
touch src/emotional_processor/storage/__init__.py
touch src/emotional_processor/retrieval/__init__.py
touch src/emotional_processor/utils/__init__.py
touch src/emotional_processor/api/__init__.py
touch src/emotional_processor/api/endpoints/__init__.py
touch src/emotional_processor/api/middleware/__init__.py
touch src/emotional_processor/config/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
touch tests/performance/__init__.py
touch tests/e2e/__init__.py
touch tests/fixtures/__init__.py
touch scripts/__init__.py
echo "  ✅ Created __init__.py files"

echo ""
echo "✅ File organization complete!"
echo ""
echo "📊 Summary of moved files:"
echo "  • Core models and processors -> src/emotional_processor/"
echo "  • Scripts -> scripts/"
echo "  • Configuration -> config/"
echo "  • Tests -> tests/"
echo "  • Docker files -> docker/"
echo ""
echo "🚀 Next steps:"
echo "  1. Install dependencies: pip install -r requirements-dev.txt"
echo "  2. Start Qdrant: docker-compose -f docker/docker-compose.yml up -d qdrant"
echo "  3. Initialize database: python scripts/setup_database.py"
echo "  4. Run tests: pytest"
echo "  5. Process conversations: python scripts/process_conversation_enhanced.py your_file.json"
echo ""
echo "🔍 Run 'ls -la src/emotional_processor/' to verify the structure"
