# Note: This is a simplified, "cartoonified" representation for educational purposes.
class BaseLM:
    """Base class that defines what any language model wrapper must do"""
    def __init__(self, model, model_type='chat', temperature=0.0,
                 max_tokens=1000, cache=True, **kwargs):
        # Common setup for any LM implementation
        self.model = model
        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self.kwargs = kwargs

    def __call__(self, prompt=None, messages=None, **kwargs):
        # This must be implemented by subclasses
        raise NotImplementedError("This method should be overridden by subclasses.")
