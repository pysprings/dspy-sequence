"""Real-world signature (§4): the sline case-study ShellAssistant.

Contract carries context fields, behavioral rules, and output constraints —
all inside a structured docstring.
"""

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=200))


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


assistant = dspy.Predict(ShellAssistant)

for request in [
    "show me the 5 largest files here",
    "grpe for TODO in python files",  # typo the LM should fix
    "compress this directory into a tarball named backup.tar.gz",
]:
    result = assistant(
        shell="zsh",
        os_name="Linux",
        cwd="/home/ryan/src/dspy-sequence",
        request=request,
    )
    print(f"> {request}")
    print(f"  {result.command}\n")

# Uncomment to see the full compiled prompt + raw LM response for the last call:
# dspy.inspect_history(n=1)
