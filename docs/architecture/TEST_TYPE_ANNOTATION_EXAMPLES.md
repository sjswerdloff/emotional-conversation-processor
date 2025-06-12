# Test Type Annotation Examples

Based on the Testing and Debugging Methodology document, here are the correct type annotations for test functions in `test_replayer.py`:

## Basic Test Function Signatures

### 1. Simple test with fixture

```python
def test_init_with_api_key(self, anthropic_api_key: str) -> None:
    """Test initialization with API key"""
    # test implementation
```

### 2. Async test with Path fixture

```python
async def test_complete_load_and_validate_workflow(self, temp_conversation_file: Path) -> None:
    """Test complete workflow from initialization to loaded conversation"""
    # test implementation
```

### 3. Test with multiple fixtures

```python
async def test_model_upgrade_workflow(
    self,
    mock_anthropic_class: MagicMock,  # from @patch decorator
    valid_conversation_data: dict[str, Any]
) -> None:
    """Test workflow with model upgrade from original conversation"""
    # test implementation
```

### 4. Test with no parameters

```python
def test_init_without_api_key(self) -> None:
    """Test initialization fails without API key"""
    # test implementation
```

### 5. Setup method

```python
def setup_method(self) -> None:
    """Setup method for test class"""
    self.api_key = "test-api-key"
    self.replayer = EmotionalConversationReplayer(api_key=self.api_key)
```

## Common Fixture Types

Based on the conftest.py file:

- `anthropic_api_key: str`
- `valid_conversation_data: dict[str, Any]`
- `temp_conversation_file: Path`
- `conversation_with_extensions: dict[str, Any]`
- `temp_invalid_conversation_file: Path`
- `temp_malformed_json_file: Path`
- `mock_anthropic_response: MagicMock`
- `capsys: pytest.CaptureFixture[str]` (pytest built-in)

## Import Requirements

Add these imports at the top of the test file:

```python
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import pytest
```

## Note on Mock Parameters

When using `@patch` decorators, the injected mock parameter should be typed as `MagicMock`. Even if not used in the test body, it's part of the function signature and should be typed.
