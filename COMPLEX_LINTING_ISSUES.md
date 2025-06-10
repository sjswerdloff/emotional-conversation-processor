# Complex Linting Issues

These linting issues require discussion or clarification before fixing:

## 1. ARG002 - Unused mock in test decorator (test_replayer.py:407)

```python
@patch("anthropic.Anthropic")
async def test_model_upgrade_workflow(self, mock_anthropic_class, valid_conversation_data):
```

**Issue**: The `mock_anthropic_class` parameter is injected by the `@patch` decorator but not used in the test body.

**Complexity**: This is a common pattern in testing where the mock is needed to prevent actual API calls even if not directly referenced. Removing it would break the decorator's function signature expectations.

**Options**:

1. Add `# noqa: ARG002` to suppress the warning
2. Use the mock in a minimal way (e.g., assert it was called)
3. Restructure the test to avoid the patch decorator

## 2. Missing Type Annotations in Test Functions ✅ RESOLVED

~~While the linter auto-fixed many issues, the Test Writing Philosophy document emphasizes that ALL test functions should have type annotations including return type `-> None`. The current tests are missing these annotations.~~

**Status**: FIXED - Added type annotations to all test functions in:

- `test_model_manager.py`
- `test_conversation_validator.py`
- `test_replayer.py`
- Various script files

## 3. Batch Size Parameter (process_conversation.py:189)

The `batch_size` parameter is documented but not used in the implementation:

```python
def process_conversation_file(
    self, file_path: str, conversation_id: str | None = None, batch_size: int = 50  # noqa: ARG002
) -> list[ConversationSegment]:
```

**Complexity**: This appears to be a placeholder for future functionality. Should we:

1. Keep it with the noqa comment (current solution)
2. Remove it entirely
3. Implement the batching functionality

## Status Summary

✅ **Resolved**: Type annotations have been added to all test functions

⚠️ **Still Complex**:

1. Mock parameter in test decorator - recommend `# noqa: ARG002`
2. Batch size parameter - clarify if implementation is planned

## Recommendations

1. For the test mock issue, I recommend using `# noqa: ARG002` as it's a legitimate testing pattern
2. ~~For type annotations in tests, consider a separate PR to add them systematically~~ ✅ COMPLETED
3. For the batch_size parameter, clarify if batching is planned or if the parameter should be removed
