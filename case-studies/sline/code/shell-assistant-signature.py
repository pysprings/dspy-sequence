"""DSPy Signature and Module for shell command assistance.

This file demonstrates how to express sline's prompt as a DSPy Signature,
making the implicit contract explicit and enabling optimization.
"""

import dspy


class ShellAssistant(dspy.Signature):
    """You are a shell command assistant. Given a natural language description 
    or partial command, return ONLY a valid shell command. No explanation, no 
    markdown, no code fences - just the raw command.

    Context:
    - Shell: {shell}
    - OS: {os_name}
    - Working directory: {cwd}

    Rules:
    1. Output exactly one command (may use pipes, &&, etc.)
    2. Use syntax appropriate for the specified shell
    3. Prefer common utilities available on the OS
    4. If the input is already a valid command with a typo, fix it
    5. If unclear, make a reasonable assumption"""
    
    shell: str = dspy.InputField()
    os_name: str = dspy.InputField()
    cwd: str = dspy.InputField()
    request: str = dspy.InputField()
    
    command: str = dspy.OutputField()


class ShellAssistantModule(dspy.Module):
    """DSPy module wrapping the shell assistant signature.
    
    This is the standard pattern: wrap a Signature in a Module to enable
    optimization and composition with other modules.
    """
    
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ShellAssistant)
    
    def forward(self, shell: str, os_name: str, cwd: str, request: str) -> dspy.Prediction:
        return self.predict(
            shell=shell,
            os_name=os_name,
            cwd=cwd,
            request=request
        )


# Example usage
if __name__ == "__main__":
    import os
    
    # Configure DSPy with OpenRouter
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY to run this example")
        exit(1)
    
    lm = dspy.LM(
        model="openrouter/openai/gpt-oss-20b",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )
    dspy.configure(lm=lm)
    
    # Create the module
    assistant = ShellAssistantModule()
    
    # Test it
    result = assistant(
        shell="zsh 5.9",
        os_name="Linux Mint 22.1",
        cwd="/home/user/projects",
        request="list files by size"
    )
    
    print(f"Request: list files by size")
    print(f"Command: {result.command}")
