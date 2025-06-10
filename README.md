# Emotional Conversation Processor

A sophisticated system for processing and replaying emotional context from LLM conversations using vector databases. This tool preserves the emotional arc and relational dynamics of extended conversations while filtering out granular technical problem-solving details.

## Features

- **Emotional Context Preservation**: Advanced emotion classification to identify and preserve emotionally significant conversation segments
- **Technical Content Filtering**: Intelligent detection and deprioritization of technical problem-solving content
- **Vector-Based Retrieval**: Uses Qdrant vector database for efficient semantic search and context retrieval
- **Conversation Replay**: Reconstructs conversations maintaining emotional continuity and key discussion points
- **Extensible Architecture**: Modular design allowing for custom embeddings, retrieval strategies, and processing pipelines

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Docker (for Qdrant database)
- At least 4GB RAM for embedding models

### Installation

1. **Clone and set up the project**:

```bash
git clone <your-repo-url>
cd emotional-conversation-processor
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:

```bash
pip install -r requirements-dev.txt
```

3. **Set up the database**:

```bash
# Start Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant

# Initialize the database
python scripts/setup_database.py
```

4. **Download required models**:

```bash
python -c "import spacy; spacy.cli.download('en_core_web_sm')"
```

### Basic Usage

1. **Process a conversation**:

```python
from emotional_processor import process_conversation_history

# Process your conversation file
vector_store = process_conversation_history("path/to/your/conversation.txt")
```

2. **Replay with emotional context**:

```python
from emotional_processor.retrieval import EmotionalConversationReplayer

replayer = EmotionalConversationReplayer(vector_store)
context = replayer.retrieve_emotional_context(
    current_message="I'm feeling grateful for our collaboration",
    context_limit=5
)
```

3. **Command-line processing**:

```bash
# Process a conversation file
ecp-process --input conversations/my_chat.txt --output data/processed/

# Check system health
ecp-health --verbose
```

## Architecture

The system consists of several key components:

- **Emotion Classification**: Uses transformer models to identify emotional content
- **Technical Detection**: Pattern-based filtering of technical/problem-solving content
- **Embedding Generation**: Creates emotion-aware embeddings for semantic search
- **Vector Storage**: Qdrant database for efficient similarity search
- **Context Retrieval**: Intelligent selection of emotionally relevant conversation history

## Configuration

Configuration is managed through YAML files in the `config/` directory:

- `development.yaml`: Development environment settings
- `production.yaml`: Production deployment configuration
- `testing.yaml`: Test environment configuration

Key configuration options:

```yaml
# config/development.yaml
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384

emotion_classification:
  model: "j-hartmann/emotion-english-distilroberta-base"
  threshold: 0.4

qdrant:
  host: "localhost"
  port: 6333
  collection_name: "conversation_history"
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Run with coverage
pytest --cov=src/emotional_processor --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Development Workflow

1. Create a feature branch
2. Write tests for new functionality
3. Implement the feature
4. Ensure all tests pass and code quality checks pass
5. Submit a pull request

## API Reference

### Core Classes

#### `ConversationSegment`

```python
@dataclass
class ConversationSegment:
    content: str
    speaker: str
    timestamp: Optional[str]
    emotional_score: float
    emotional_labels: List[str]
    technical_score: float
    importance_weight: float
```

#### `EmotionalConversationReplayer`

```python
class EmotionalConversationReplayer:
    def retrieve_emotional_context(
        self,
        current_message: str,
        context_limit: int = 5,
        emotional_weight: float = 0.7
    ) -> List[Dict]
```

### Processing Pipeline

```python
# Complete processing pipeline
def process_conversation_history(conversation_file_path: str) -> ConversationVectorStore:
    """Process conversation file and return configured vector store."""
```

## Performance Considerations

- **Memory Usage**: Embedding models require 1-2GB RAM
- **Processing Speed**: ~100 conversation segments per minute
- **Storage**: Vector embeddings require ~1.5KB per conversation segment
- **Retrieval**: Sub-second semantic search with Qdrant

## Troubleshooting

### Common Issues

1. **Qdrant Connection Error**: Ensure Docker container is running on port 6333
2. **Model Download Failures**: Check internet connection and disk space
3. **Memory Issues**: Reduce batch size in configuration
4. **Slow Processing**: Consider GPU acceleration for embedding generation

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](docs/contributing.md) for details on our code of conduct and development process.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run full test suite
pytest --cov=src/emotional_processor
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Qdrant](https://qdrant.tech/) for the vector database
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [Hugging Face Transformers](https://huggingface.co/transformers/) for emotion classification
- [spaCy](https://spacy.io/) for natural language processing

## Support

- **Documentation**: [Full documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/emotional-conversation-processor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/emotional-conversation-processor/discussions)

## Roadmap

- [ ] Multi-language emotion classification
- [ ] Custom emotion model training
- [ ] Real-time conversation processing
- [ ] Web interface for conversation management
- [ ] Advanced retrieval strategies
- [ ] Performance optimizations for large datasets
