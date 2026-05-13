"""Abstract base for adapters that encode the entire interaction in a
single structured format (YAML, HTML, JSON, XML, ...).

Inherits directly from ``dspy.adapters.base.Adapter`` -- the bare
abstract base, not ``ChatAdapter``. The base ``Adapter`` declares six
abstract methods. This class fills in five of them with a generic
single-format encoding strategy and leaves three small hooks --
``shape``, ``encode``, ``decode`` -- for subclasses to fill in with the
format-specific bits.

Compare with the ``SlineAdapter`` in the case study: sline overrides
``Adapter.format()`` itself to rewrite the entire prompt shape, so it
doesn't fit this family. Two kinds of customization, two patterns.
"""

from dspy.adapters.base import Adapter


def list_fields(signature):
    """Render a signature's input and output fields as a numbered list."""

    def render(fields):
        items = []
        for i, (name, info) in enumerate(fields.items(), 1):
            desc = (info.json_schema_extra or {}).get("desc", "") or ""
            # DSPy uses '${name}' as a synthetic placeholder when no desc is
            # supplied; treat that as "no description" so the prompt stays clean.
            if desc == f"${{{name}}}":
                desc = ""
            line = f"  {i}. `{name}` ({info.annotation.__name__})"
            if desc:
                line += f": {desc}"
            items.append(line)
        return "\n".join(items)

    return (
        f"Your input fields are:\n{render(signature.input_fields)}\n\n"
        f"Your output fields are:\n{render(signature.output_fields)}"
    )


class StructuredOutputAdapter(Adapter):
    """Adapter whose entire interaction is in a single structured format.

    Subclasses set ``name`` and implement ``shape``, ``encode``, ``decode``.
    """

    name: str = ""

    # ---- All six Adapter responsibilities, stated explicitly ----

    def format_field_description(self, signature):
        return list_fields(signature)

    def format_field_structure(self, signature):
        # Leading blank line: base.format_system_message joins blocks with a
        # single '\n', so an extra newline here separates this block from the
        # field-description block above it.
        return (
            f"\nInputs come as {self.name}:\n\n"
            f"{self.shape(signature.input_fields)}\n\n"
            f"Respond in {self.name}:\n\n"
            f"{self.shape(signature.output_fields)}"
        )

    def format_task_description(self, signature):
        return f"\nTask: {signature.instructions}"

    def format_user_message_content(
        self, signature, inputs, prefix="", suffix="", main_request=False
    ):
        payload = {k: inputs[k] for k in signature.input_fields if k in inputs}
        body = self.encode(payload)
        return "\n\n".join(part for part in (prefix, body, suffix) if part).strip()

    def format_assistant_message_content(
        self, signature, outputs, missing_field_message=None
    ):
        payload = {
            k: outputs.get(k, missing_field_message or "")
            for k in signature.output_fields
        }
        return self.encode(payload)

    def parse(self, signature, completion):
        return self.decode(completion, list(signature.output_fields))

    # ---- The three things subclasses customize ----

    def shape(self, fields):
        """Render placeholder scaffolding for the given fields."""
        raise NotImplementedError

    def encode(self, values):
        """Encode a dict of values to the wire format."""
        raise NotImplementedError

    def decode(self, completion, output_names):
        """Parse the wire format back into a dict of output values."""
        raise NotImplementedError
