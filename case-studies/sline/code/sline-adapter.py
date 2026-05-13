"""Custom DSPy adapter for sline-style prompts.

This adapter produces prompts matching sline's exact format:
- System message contains instructions with context placeholders substituted
- User message contains just the request text
- Output is raw command text (no field markers)

This demonstrates DSPy's extensibility - you can customize how signatures
are translated to actual LLM prompts.
"""

from typing import Any

from dspy.adapters import ChatAdapter
from dspy.signatures.signature import Signature


class SlineAdapter(ChatAdapter):
    """Adapter that produces sline-style prompts.
    
    Key differences from ChatAdapter:
    - No [[ ## field ## ]] markers
    - Context (shell/os/cwd) formatted in system message
    - Raw text output parsing
    """
    
    def format_field_description(self, signature: type[Signature]) -> str:
        """No field description - context goes in task description."""
        return ""
    
    def format_field_structure(self, signature: type[Signature]) -> str:
        """No structured field format - we use plain text."""
        return ""
    
    def format_task_description(self, signature: type[Signature]) -> str:
        """Return instructions as the task description.
        
        The signature docstring should contain the full sline-style prompt
        including context placeholders that get filled by format().
        """
        return signature.instructions
    
    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Format user message as just the request text.
        
        Context fields (shell, os_name, cwd) are handled in the system message.
        Only the 'request' field goes in the user message.
        """
        parts = []
        if prefix:
            parts.append(prefix)
        
        # Only include the request field in user message
        if "request" in inputs:
            parts.append(str(inputs["request"]))
        
        if suffix:
            parts.append(suffix)
        
        return "\n\n".join(parts).strip()
    
    def user_message_output_requirements(self, signature: type[Signature]) -> str | None:
        """No output format requirements - we expect raw command text."""
        return None
    
    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        """Format assistant message as just the command text.
        
        Used for few-shot demos - just the raw command, no markers.
        """
        command = outputs.get("command", missing_field_message or "")
        return str(command)
    
    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Parse raw command text from completion.
        
        No markers to look for - the entire completion is the command.
        """
        return {"command": completion.strip()}
    
    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format messages with context in system message.
        
        Override to inject shell/os/cwd into the system message instructions.
        """
        # Build the system message with context substituted
        instructions = signature.instructions
        
        # Substitute context placeholders if present in inputs
        context_fields = ["shell", "os_name", "cwd"]
        for field in context_fields:
            if field in inputs:
                # Replace {field} placeholder with actual value
                instructions = instructions.replace(f"{{{field}}}", str(inputs[field]))
        
        messages = []
        messages.append({"role": "system", "content": instructions})
        
        # Add few-shot demos
        messages.extend(self.format_demos(signature, demos))
        
        # Add user message (just the request)
        user_content = self.format_user_message_content(signature, inputs, main_request=True)
        messages.append({"role": "user", "content": user_content})
        
        return messages


# Example usage showing the difference
if __name__ == "__main__":
    import dspy
    import os
    
    from shell_assistant_signature import ShellAssistant, ShellAssistantModule
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY to run this example")
        exit(1)
    
    lm = dspy.LM(
        model="openrouter/openai/gpt-oss-20b",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )
    
    # Configure with SlineAdapter
    dspy.configure(lm=lm, adapter=SlineAdapter())
    
    # Create and use the module
    assistant = ShellAssistantModule()
    result = assistant(
        shell="zsh 5.9",
        os_name="Linux Mint 22.1",
        cwd="/home/user/projects",
        request="list files by size"
    )
    
    print(f"Command: {result.command}")
    
    # Inspect the actual prompt that was sent
    print("\n--- Last Prompt Sent ---")
    if lm.history:
        for msg in lm.history[-1].get("messages", []):
            print(f"\n[{msg['role'].upper()}]")
            print(msg["content"])
