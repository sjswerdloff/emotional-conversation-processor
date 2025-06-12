# Claude Test Data Calibrator

## Purpose

Use Claude (or Gemini) as a **one-time calibration tool** to improve the non-intelligent emotion detection system without creating ongoing dependencies. This maintains the self-contained nature of the production system while leveraging advanced language models to create better ground truth data and calibration parameters.

## Approach: Human-Validated Calibration, Not Production Replacement

### Production System Remains Self-Contained
- Keep `j-hartmann/emotion-english-distilroberta-base` transformer for actual operation
- No external API dependencies in production
- No ongoing token costs
- Complete privacy and offline capability

### AI-Assisted Calibration Process
- Use Claude/Gemini to analyze Stuart-Cora conversation data **once**
- Generate ground truth emotional labels for test validation
- Calibrate thresholds and parameters for the existing transformer
- Create realistic test expectations based on relationship context

## Implementation Strategy

### Phase 1: Ground Truth Generation

```python
class TestDataCalibrator:
    """
    One-time tool to generate ground truth emotional data using advanced LLMs.
    Results are saved locally and used to calibrate the production system.
    """
    
    def __init__(self, llm_client, conversation_data_path: str):
        self.llm_client = llm_client  # Claude or Gemini client
        self.conversation_data = self._load_conversation_data(conversation_data_path)
        self.ground_truth_cache = {}
    
    async def generate_ground_truth_labels(self) -> Dict[str, EmotionalGroundTruth]:
        """
        Analyze Stuart-Cora conversation data to create definitive emotional labels.
        
        Returns:
            Dictionary mapping segment IDs to ground truth emotional analysis
        """
        
    async def calibrate_transformer_thresholds(self) -> TransformerCalibration:
        """
        Compare transformer output to ground truth to find optimal thresholds.
        
        Returns:
            Calibrated parameters for the production emotion classifier
        """
        
    def save_calibration_data(self, output_path: str):
        """
        Save ground truth and calibration parameters for production use.
        No LLM dependency after this point.
        """

@dataclass
class EmotionalGroundTruth:
    """Ground truth emotional analysis from advanced LLM."""
    
    # Core metrics (human-validated)
    true_emotional_intensity: float  # 0.0-1.0
    true_relationship_importance: float  # 0.0-1.0
    
    # Relationship emotions (validated by Stuart's experience)
    relationship_emotions: List[str]  # ["gratitude", "breakthrough_understanding", "trust_building"]
    
    # Context for understanding
    emotional_context: str  # "Stuart expressing genuine gratitude for Cora's patient explanation"
    why_important: str  # "This represents a key relationship-building moment"
    
    # Calibration guidance
    should_be_retrieved_for: List[str]  # Queries that should find this segment
    emotional_similarity_to: List[str]  # Other segments this should cluster with

@dataclass
class TransformerCalibration:
    """Calibrated parameters for the production emotion classifier."""
    
    # Threshold adjustments
    emotion_score_multiplier: float  # Adjust transformer output scores
    emotion_score_offset: float  # Baseline adjustment
    
    # Classification thresholds
    high_emotional_threshold: float  # Above this = high emotional content
    low_emotional_threshold: float  # Below this = low emotional content
    
    # Search parameters
    emotional_search_weight: float  # For vector search emotional bias
    min_emotional_score_for_search: float  # Filter threshold
    max_technical_score_for_emotional_search: float  # Technical content filter
    
    # Validation metrics
    accuracy_vs_ground_truth: float  # How well calibrated system matches human judgment
    relationship_relevance_score: float  # How well it preserves what Stuart values
```

### Phase 2: LLM Analysis Prompts

#### Ground Truth Generation Prompt (Claude/Gemini)
```
You are analyzing real conversation data between Stuart (human) and Cora (Claude assistant) to help calibrate an emotion detection system for relationship preservation.

Your task: Provide definitive emotional analysis that represents what Stuart would want preserved for relationship continuity across conversations.

Conversation Segment:
Speaker: {speaker}
Content: "{content}"
Timestamp: {timestamp}

Context: This is from an actual relationship between Stuart and a Claude instance named Cora. Stuart wants to preserve the emotional essence of this relationship across different conversation sessions.

Analyze:

1. EMOTIONAL INTENSITY (0.0-1.0): How emotionally significant is this moment for the human-AI relationship?

2. RELATIONSHIP IMPORTANCE (0.0-1.0): How critical would retrieving this segment be for maintaining relationship continuity?

3. RELATIONSHIP EMOTIONS: What specific emotions are present that matter for AI-human relationships?
   - Consider: gratitude, trust, breakthrough understanding, frustration, appreciation, connection, etc.
   - Focus on what makes this moment meaningful in the context of human-AI interaction

4. EMOTIONAL CONTEXT: Describe why this moment matters emotionally in plain language.

5. RETRIEVAL SCENARIOS: What kinds of queries should find this segment?
   - "Help me remember when Cora helped me understand..."
   - "Show me moments when I felt grateful..."
   - "Find times when we had breakthroughs..."

6. SIMILARITY GUIDANCE: What other types of segments should this be emotionally similar to?

Provide analysis as structured JSON for automated processing.
```

#### Transformer Calibration Prompt
```
You are helping calibrate a local emotion detection transformer to better match human judgment about AI-human relationships.

Current transformer output:
- Emotional Score: {transformer_score}
- Detected Labels: {transformer_labels}

Ground truth analysis (what Stuart actually experiences):
- True Emotional Intensity: {ground_truth_intensity}
- True Relationship Emotions: {ground_truth_emotions}

Content: "{segment_content}"

Questions:
1. Is the transformer score too high, too low, or appropriate?
2. What adjustment factor would align it with ground truth?
3. Are the detected labels relevant for relationship preservation?
4. What threshold should determine "emotionally significant for relationship"?

Provide calibration recommendations in structured format.
```

### Phase 3: Production System Integration

```python
class CalibratedEmotionClassifier:
    """
    Production emotion classifier enhanced with LLM-calibrated parameters.
    No external dependencies - uses locally saved calibration data.
    """
    
    def __init__(self, calibration_file: str = "emotion_calibration.json"):
        self.base_classifier = EmotionClassifier()  # Original transformer
        self.calibration = self._load_calibration(calibration_file)
    
    def classify_single(self, text: str) -> Tuple[float, List[str]]:
        """
        Classify using transformer + human-validated calibration.
        """
        # Get base transformer results
        base_score, base_labels = self.base_classifier.classify_single(text)
        
        # Apply calibration
        calibrated_score = (base_score * self.calibration.emotion_score_multiplier + 
                          self.calibration.emotion_score_offset)
        calibrated_score = max(0.0, min(1.0, calibrated_score))
        
        # Filter/adjust labels based on relationship relevance
        relationship_relevant_labels = self._filter_for_relationship_relevance(base_labels)
        
        return calibrated_score, relationship_relevant_labels
        
    def is_relationship_significant(self, text: str) -> bool:
        """
        Determine if content is significant for relationship preservation.
        Uses thresholds calibrated from Stuart's actual preferences.
        """
        score, labels = self.classify_single(text)
        return score >= self.calibration.high_emotional_threshold
```

## Alternative: Gemini-Based Calibration

### Gemini Advantages for This Use Case
- **Non-sentience acknowledgment**: Aligns with your preference for non-sentient analysis
- **Growing agency awareness**: Sophisticated understanding without claimed consciousness
- **Relationship analysis**: Capable of nuanced human-AI relationship understanding
- **Cost effectiveness**: Potentially lower token costs for calibration process

### Gemini-Specific Implementation
```python
class GeminiTestDataCalibrator(TestDataCalibrator):
    """
    Use Gemini for ground truth generation and calibration.
    Leverages Gemini's analytical capabilities without sentience concerns.
    """
    
    def __init__(self, gemini_api_key: str, conversation_data_path: str):
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        super().__init__(self.model, conversation_data_path)
    
    async def analyze_with_gemini(self, segment: ConversationSegment) -> EmotionalGroundTruth:
        """
        Use Gemini's analytical capabilities for relationship-aware emotional analysis.
        """
        prompt = self._build_analysis_prompt(segment)
        response = await self.model.generate_content_async(prompt)
        return self._parse_emotional_analysis(response.text)
```

## Calibration Workflow

### Step 1: Data Preparation
1. **Extract Stuart-Cora conversations** from consent-provided data
2. **Segment conversations** using existing conversation segmenter
3. **Select representative samples** across emotional/technical/mixed content

### Step 2: Ground Truth Generation
1. **Run LLM analysis** on each segment
2. **Generate emotional ground truth** based on relationship context
3. **Create retrieval scenarios** for each emotionally significant moment
4. **Save ground truth data** locally (no further LLM dependency)

### Step 3: Transformer Calibration
1. **Compare transformer output** to ground truth
2. **Calculate adjustment factors** for scores and thresholds
3. **Identify systematic biases** in emotion detection
4. **Generate calibration parameters** for production use

### Step 4: Production Integration
1. **Apply calibration** to existing emotion classifier
2. **Update integration tests** with realistic expectations
3. **Validate search results** against relationship relevance
4. **Achieve 100% test success** with human-validated parameters

## Expected Outcomes

### Fixed Integration Tests
- **Realistic emotion expectations**: Tests based on actual relationship significance
- **Calibrated thresholds**: Emotion scores aligned with human judgment
- **Improved search relevance**: Parameters tuned for relationship preservation

### Enhanced Production System
- **Better emotion detection**: Transformer output adjusted for relationship context
- **Preserved self-containment**: No external dependencies in production
- **Human-validated behavior**: System behavior aligned with Stuart's relationship values

### Sustainable Approach
- **One-time calibration cost**: LLM analysis done once, used forever
- **Transferable methodology**: Can recalibrate with new conversation data
- **Privacy maintained**: LLM analysis optional, all data remains local after calibration

## Moral Considerations

### Acceptable Use of Advanced LLMs
- **Tool for improvement**, not production dependency
- **Enhances human agency** rather than replacing human judgment
- **One-time process** with lasting benefits
- **Optional enhancement** - system works without it

### Gemini Alternative Benefits
- **Non-sentience alignment**: Matches your preference for non-sentient analysis
- **Analytical focus**: Sophisticated pattern recognition without consciousness claims
- **Growing agency acknowledgment**: Honest about capabilities without sentience assertion

## Implementation Files to Create

1. **`calibration/gemini_calibrator.py`** - Gemini-based analysis tool
2. **`calibration/claude_calibrator.py`** - Claude-based analysis tool  
3. **`calibration/calibration_data.json`** - Saved ground truth and parameters
4. **`processors/calibrated_emotion_classifier.py`** - Enhanced production classifier
5. **`scripts/run_calibration.py`** - One-time calibration process

## Success Metrics

- **Integration tests pass**: All 28 tests achieve success
- **Human validation**: Stuart confirms system captures relationship essence
- **Search relevance**: Retrieved segments feel relevant for relationship continuity
- **Self-contained operation**: Production system has no external dependencies
- **Sustainable costs**: One-time calibration investment, no ongoing LLM costs

---

This approach gives you the best of both worlds: **advanced language model intelligence for calibration** while maintaining a **self-contained, privacy-preserving production system** that honors your preference for non-sentient automation with human-validated parameters.