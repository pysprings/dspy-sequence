# Session 1: LM Setup - Gateway to AI Services
*DSPy Mastery Series - Month 1 - July 2025*

## 1. Opening & Context Setting (10 mins)

Welcome to the DSPy Mastery Series! We're embarking on a 12-month journey to
master systematic AI development using Python patterns you already know. Today
we start with the foundation: the LM (Language Model) class.

**Why LM matters:** Think of `dspy.LM` like the `requests` library for AI
models - it's a unified interface that lets you talk to any AI provider using
the same Python code. Just like `requests.get()` works whether you're calling
GitHub's API or a weather service, `dspy.LM` works whether you're using OpenAI,
Anthropic, or local models.

**Session Goals:**

- Understand LM as your gateway to AI services
- Grasp the architecture without getting lost in complexity

## 2. Hello World with DSPy LM (15 mins)

Let's start with an example.

```python
import dspy

# Create a language model instance
lm = dspy.LM('openrouter/openai/gpt-4o-mini', api_key='your-openrouter-key')

# Configure DSPy to use this model globally
dspy.configure(lm=lm)

# Direct usage - simple prompt
response = lm("What is the capital of France?")
print(response)
```

**Key Takeaway:** DSPy makes AI integration as simple as any other Python
library - no complex setup, no vendor-specific APIs to learn.

## 3. Unpacking the LM Class (20 mins)

Now let's understand what's happening under the hood, using some "cartoonified code":

### The LM Class (Simplified)
```python
class LM:
    def __init__(self, model: str, model_type='chat', temperature=0.0, 
                 max_tokens=4000, cache=True, num_retries=3, **kwargs):
        # Store the model identifier (like "openai/gpt-4o-mini")
        self.model = model
        self.model_type = model_type  # 'chat' or 'text'
        self.cache = cache
        self.num_retries = num_retries
        self.history = []  # Track all your API calls
        
        # Combine all parameters for the API call
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
    
    def __call__(self, prompt=None, messages=None, **kwargs):
        """This is what happens when you call lm('your prompt')"""
        # Convert prompt to messages format if needed
        messages = messages or [{"role": "user", "content": prompt}]
        
        # Merge any new parameters
        merged_kwargs = {**self.kwargs, **kwargs}
        
        # Make the actual API call (with retries and caching)
        response = self.forward(
            model=self.model, 
            messages=messages, 
            **merged_kwargs
        )
        
        # Process response and update history
        outputs = self._extract_text_from_response(response)
        self.history.append({
            "prompt": prompt,
            "messages": messages, 
            "response": response,
            "outputs": outputs
        })
        
        return outputs
    
    def forward(self, **request_params):
        """The actual API call - this is where LiteLLM does the work"""
        return litellm.completion(**request_params)
    
    def copy(self, **new_params):
        """Create a new LM with different settings"""
        # Copy this LM but change some parameters
        return LM(self.model, **{**self.__dict__, **new_params})
```

### The Foundation: BaseLM
The LM class actually inherits from `BaseLM`, which provides the common interface:

```python
class BaseLM:
    """Base class that defines what any language model wrapper must do"""
    def __init__(self, model, model_type='chat', temperature=0.0, 
                 max_tokens=1000, cache=True, **kwargs):
        # Common setup for any LM implementation
        pass
    
    def __call__(self, prompt=None, messages=None, **kwargs):
        # This must be implemented by subclasses
        pass
```

### Key Parameters Explained
| Parameter | Default | What It Does |
|-----------|---------|-------------|
| `model` | required | Model identifier like `"openai/gpt-4o-mini"` |
| `temperature` | 0.0 | Controls randomness (0.0 = deterministic) |
| `max_tokens` | 4000 | Maximum response length |
| `cache` | True | Saves money by caching identical requests |

**Where LM Lives in DSPy:** Every DSPy module (`dspy.Predict`,
`dspy.ChainOfThought`, etc.) uses the configured LM under the hood. It's the
universal translator between your Python code and AI services.

## 4. The OpenRouter Ecosystem (15 mins)

OpenRouter is like a universal broker for AI models - one API to access
hundreds of AI models through a single endpoint. Think of it as the "Amazon for
AI models" - instead of managing subscriptions and APIs for each vendor service
(provider), you get everything in one place.

### Why OpenRouter Matters for Python Developers

**The Problem Without OpenRouter:**
```python
# Without OpenRouter - managing multiple APIs
import openai
import anthropic
from google.cloud import aiplatform

# Different setup for each provider
openai_client = openai.OpenAI(api_key="sk-...")
anthropic_client = anthropic.Anthropic(api_key="sk-ant-...")
# ... different authentication, different parameters, different error handling
```

**The Solution With OpenRouter:**
```python
# With OpenRouter - one API for everything
import dspy

# All models through OpenRouter's unified interface
openai_lm = dspy.LM('openrouter/openai/gpt-4o-mini')
anthropic_lm = dspy.LM('openrouter/anthropic/claude-3-haiku-20240307')
google_lm = dspy.LM('openrouter/google/gemini-1.5-flash')
meta_lm = dspy.LM('openrouter/meta-llama/llama-3.2-3b-instruct')

# Local models still use direct provider format
local_lm = dspy.LM('ollama/llama3.2:1b')
```

**Note:** DSPy supports both approaches - you can use models directly from
providers (e.g., `openai/gpt-4o-mini`) or route them through OpenRouter (e.g.,
`openrouter/openai/gpt-4o-mini`) for additional benefits like automatic
failover and cost optimization.

### Core Value Propositions

**1. Effortless Model Switching**
OpenRouter removes the need for code modifications when switching AI providers
because developers can keep using its API after switching to a new AI model:

```python
# Start with a fast, cheap model for development
dspy.configure(lm=dspy.LM('openrouter/openai/gpt-4o-mini'))

# Switch to a more powerful model for production
# Just change the string - no code changes needed!
dspy.configure(lm=dspy.LM('openrouter/anthropic/claude-3-5-sonnet-20241022'))

# Or even switch to a local model for privacy
dspy.configure(lm=dspy.LM('ollama/llama3.2:3b'))
```

**2. Automatic Failover & Reliability**
OpenRouter provides reliable AI models via distributed infrastructure with
automatic fallbacks to other providers when one goes down. Your code keeps
working even if OpenAI has an outage:

```python
# If OpenAI goes down, automatically switches to another provider
lm = dspy.LM('openrouter/mistralai/mistral-small-24b-isnstruct-2501')  # Transparent failover built-in
```

**3. Cost Optimization Features**
Developers can configure the platform to prioritize the most cost-efficient
LLMs, with caching features that reuse common prompt responses:

```python
# Use cost-optimized routing
lm = dspy.LM('openrouter/meta-llama/llama-3.2-3b-instruct:floor')  # :floor = cheapest

# Free models for experimentation
free_lm = dspy.LM('openrouter/meta-llama/llama-3.2-3b-instruct:free')

# Speed-optimized routing
fast_lm = dspy.LM('openrouter/openai/gpt-4o-mini:nitro')  # :nitro = fastest
```

### Model Variant Magic

OpenRouter supports special suffixes that can be added to model names to change their behavior:

| Variant | Purpose | Example |
|---------|---------|---------|
| `:free` | Free tier with rate limits | `llama-3:free` |
| `:floor` | Cheapest available provider | `gpt-4o:floor` |
| `:nitro` | Fastest response times | `claude-3-haiku:nitro` |
| `:online` | Web search capabilities | `gpt-4o:online` |

### Spectrum of Capabilities

**Speed Demons (Fast & Cheap):**

- `openrouter/openai/gpt-4o-mini` - OpenAI's efficient model
- `openrouter/anthropic/claude-3-haiku-20240307` - Anthropic's fastest
- `openrouter/google/gemini-1.5-flash` - Google's speed champion

**Powerhouses (Complex Reasoning):**

- `openrouter/openai/o3` - OpenAI's flagship
- `openrouter/anthropic/claude-3-5-sonnet-20241022` - Top reasoning model
- `openrouter/google/gemini-2.5-pro` - Google's advanced model

**Local Options (Privacy & Cost Control):**

- `ollama/llama3.2:1b` - Lightweight local model
- `ollama/llama3.2:3b` - Balanced local option

**Specialized Models:**

- Models fine-tuned for coding, math, specific domains
- Access to 400+ AI models through one API

### Enterprise-Grade Features

OpenRouter's enterprise edition provides expanded technical support, high rate
limits and optimizations designed to reduce AI applications' latency:

- **Unified Billing:** Aggregate billing in one place and track usage using analytics
- **BYOK Support:** Use your own provider API keys with just a 5% fee
- **Analytics Dashboard:** Track costs and usage across all models
- **Rate Limit Management:** Different rate limits for different models to share the load

### The Numbers Behind OpenRouter

OpenRouter processes 8.4 trillion tokens per month for customers, with
annualized inference spending managed by the platform topping $100 million. The
platform runs at the edge, adding just ~25ms between users and their inference.

**For Python Developers, This Means:**

- Write once, run anywhere (any AI provider)
- Built-in reliability and fallbacks
- Cost optimization without complexity
- No vendor lock-in
- Production-ready from day one

## 5. Practical Examples (20 mins)

### CLI-Based Chatbot
```python
import dspy

def build_chatbot():
    lm = dspy.LM('openrouter/openai/gpt-4o-mini')
    dspy.configure(lm=lm)
    
    print("DSPy Chatbot (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        response = lm(f"You are a helpful assistant. User says: {user_input}")
        print(f"Bot: {response[0]}")

if __name__ == "__main__":
    build_chatbot()
```

### Command Completion Example (Ctrl-L Style)
```python
def smart_completion(partial_command):
    lm = dspy.LM('openrouter/openai/gpt-4o-mini')
    
    completion_prompt = f"""
    Complete this shell command: {partial_command}
    Only return the completed command, nothing else.
    """
    
    result = lm(completion_prompt)
    return result[0].strip()

# Example usage
print(smart_completion("git commit -m "))
# Output: git commit -m "Add new feature"
```

### Using with DSPy Modules
```python
# Define a simple classifier using the LM
classifier = dspy.Predict("text -> sentiment")
result = classifier(text="I love this product!")
print(result.sentiment)  # "positive"

# Create a custom module
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.Predict("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

qa_system = SimpleQA()
answer = qa_system("What is machine learning?")
print(answer.answer)
```

**Live Coding Opportunity:** Build one of these examples with the audience, taking suggestions for modifications.

## 6. Performance & Production Considerations (10 mins)

### Caching: Your Secret Weapon
```python
# Caching enabled by default - saves money!
lm = dspy.LM('openrouter/openai/gpt-4o-mini', cache=True)

# First call - hits the API
response1 = lm("What is 2+2?")

# Second identical call - returns cached result (free!)
response2 = lm("What is 2+2?")  # No API call made
```

**Two Levels of Caching:**

- **LiteLLM-level caching:** Persistent across program runs
- **In-memory caching:** Fast LRU cache for repeated calls within session

### History Tracking
```python
lm = dspy.LM('openrouter/openai/gpt-4o-mini')
lm("Hello, world!")

# Inspect recent interactions
lm.inspect_history(n=1)

# Access programmatically
print(f"Total interactions: {len(lm.history)}")
for entry in lm.history:
    print(f"Cost: ${entry.get('cost', 'cached')}")
```

**Production Warning:** Global history can cause memory leaks. In production, consider:
```python
# Disable global history tracking
dspy.settings.disable_history = True
```

## 7. Integration Landscape (10 mins)

### LiteLLM: The Universal Translator
DSPy uses LiteLLM under the hood, which means you get:

- **100+ models** from 20+ providers
- **Unified interface** regardless of provider
- **Built-in retry logic** with exponential backoff
- **Automatic rate limiting** and error handling

### HTTP Interfaces and Streaming
```python
# Basic completion
response = lm("Tell me a story")

# Chat format
chat_response = lm(messages=[
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
])

# Streaming (coming in advanced sessions)
# for chunk in lm.stream("Long story..."):
#     print(chunk, end="")
```

### Avoiding Vendor Lock-in
The beauty of DSPy's LM abstraction:
```python
# Development with cheap model
dev_lm = dspy.LM('openrouter/openai/gpt-4o-mini')

# Production with different provider
prod_lm = dspy.LM('openrouter/anthropic/claude-3-5-sonnet-20241022')

# Same code works with both!
with dspy.context(lm=prod_lm):
    result = dspy.Predict("question -> answer")(question="Deploy!")
```

## 8. Oddities & Gotchas (5 mins)

### Model-Specific Configurations
```python
# OpenAI's reasoning models (o1, o3) have special requirements
reasoning_lm = dspy.LM(
    'openrouter/openai/o1-mini',
    temperature=1.0,  # Must be 1.0
    max_tokens=20000  # Must be >= 20,000
)
```

### Common Pitfalls:

- **API key management:** Never hardcode keys, use environment variables
- **Rate limiting:** Built-in retry handles this automatically
- **Token limits:** DSPy warns when responses are truncated
- **Model names:** Use the exact format: `"provider/model-name"`

### OpenRouter Features

- Multiple API keys for higher rate limits
- Model routing and fallbacks
- Usage analytics and cost tracking

## 9. Testing & Serialization (5 mins)

### Built-in Testing Support
```python
# Copy models with different parameters
test_lm = lm.copy(temperature=0.8, max_tokens=100)

# Inspect model state
state = lm.dump_state()
print(state)  # All configuration parameters
```

### Reproducibility
```python
# Deterministic responses for testing
deterministic_lm = dspy.LM('openrouter/openai/gpt-4o-mini', temperature=0.0)

# Same input always produces same output (when temperature=0.0)
```

## 10. Session Wrap & Next Steps (5 mins)

### Key Takeaways
**LM as Your AI Gateway:** The `dspy.LM` class abstracts away provider
complexity while giving you access to the entire AI ecosystem. It's designed to
be simple for basic use cases yet powerful enough for complex applications.

**The Three-Layer Stack:**

1. **LM Layer:** Handles provider communication (today's focus)
2. **Module Layer:** Reusable AI components (next month)
3. **Optimization Layer:** Automatic improvement (later in series)

### Preview of Month 2: Data Collection
Next month we'll explore how DSPy handles the foundation of all AI systems: data. We'll cover:

- Training data formats and collection
- Validation and test set creation
- Data preprocessing pipelines
- Quality metrics and filtering

### Questions and Collaborative Feedback
*This curriculum is being developed collaboratively - your feedback shapes the
content as we pioneer this learning path together.*

**Action Items:**

1. Set up DSPy in your local environment
2. Try the examples with your own prompts
3. Experiment with different model providers
4. Share your use cases for next session

---

**Remember:** Every expert was once a beginner. Today you've taken the first
step toward systematic AI development with Python. The LM class is your gateway
- everything else builds from here.

*Time to start wrangling some serious AI magic, Pythoneers! 🐍⚡*
