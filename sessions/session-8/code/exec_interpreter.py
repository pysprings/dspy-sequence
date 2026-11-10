"""An in-process CodeInterpreter built on exec. Unsafe by design.

DSPy's code-executing modules (ProgramOfThought, CodeAct, RLM, Flex) talk to
a CodeInterpreter: four members, listed in dspy/primitives/code_interpreter.py.

  tools       host functions the code may call
  start()     allocate the session
  execute()   run code; return FinalOutput when it calls SUBMIT(), else stdout
  shutdown()  release the session

State persists across execute() calls until shutdown(). DSPy's own
PythonInterpreter meets this contract by running Pyodide inside Deno, so
model-written code never touches your process. This class meets it with
exec(), so model-written code runs in your process with your permissions.
It exists so the session can run without installing Deno, and to show the
contract by implementing it. Do not use it outside a demo.

Two limits worth knowing: the timeout uses SIGALRM, which only works on the
main thread, so threaded runs (batch, Evaluate) get no timeout; and stdout
capture is process-wide, so threaded runs can interleave output.
"""

import ast
import contextlib
import io
import signal
import threading
from typing import Any, Callable

from dspy.primitives.code_interpreter import CodeExecutionError, FinalOutput


class _Submitted(BaseException):
    """Control flow, not an error: SUBMIT() was called."""

    def __init__(self, output: Any):
        self.output = output


def SUBMIT(output: Any) -> None:
    """The preloaded function ProgramOfThought's prompt tells the model to call.

    One required argument, as in DSPy's runner. A bare SUBMIT() is a TypeError,
    which reaches the model as an error and triggers the regenerate loop.
    """
    raise _Submitted({"output": output})


class ExecInterpreter:
    def __init__(self, tools: dict[str, Callable[..., Any]] | None = None, timeout: int = 5):
        self._tools = dict(tools or {})
        self.timeout = timeout
        self._namespace: dict[str, Any] | None = None
        self.log: list[str] = []   # every code string executed, in order, for reading back

    @property
    def tools(self) -> dict[str, Callable[..., Any]]:
        return self._tools

    def start(self) -> None:
        if self._namespace is None:
            self._namespace = {"__name__": "__generated__", "SUBMIT": SUBMIT, **self._tools}

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        self.start()
        self.log.append(code)
        self._namespace.update(variables or {})

        # REPL semantics, as in DSPy's runner: if the code ends in a bare
        # expression, its value is the output. Otherwise stdout is.
        tree = ast.parse(code, "<generated>", "exec")     # SyntaxError propagates as itself
        tail = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        body = compile(tree, "<generated>", "exec")
        last = compile(ast.Expression(tail.value), "<generated>", "eval") if tail else None

        buffer = io.StringIO()
        value = None
        with self._deadline(), contextlib.redirect_stdout(buffer):
            try:
                exec(body, self._namespace)
                if last is not None:
                    value = eval(last, self._namespace)
            except _Submitted as done:
                return FinalOutput(done.output)
            except Exception as err:
                raise CodeExecutionError(f"{type(err).__name__}: {err}") from err
        if value is not None:
            return value
        output = buffer.getvalue()
        return output if output else None

    def shutdown(self) -> None:
        self._namespace = None

    @contextlib.contextmanager
    def _deadline(self):
        if threading.current_thread() is not threading.main_thread() or not self.timeout:
            yield
            return

        def on_alarm(signum, frame):
            raise TimeoutError(f"generated code ran longer than {self.timeout}s")

        previous = signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(self.timeout)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    # The contract, exercised without a model.
    interp = ExecInterpreter(tools={"double": lambda n: 2 * n})
    print(interp.execute("x = double(21)\nprint('x is', x)"))        # stdout comes back
    print(interp.execute("x + 1"))                                     # a bare expression: its value
    print(interp.execute("SUBMIT({'answer': x})"))                     # state persisted; FinalOutput
    try:
        interp.execute("1 / 0")
    except CodeExecutionError as err:
        print("recoverable:", err)                                      # the regenerate loop sees this
    try:
        interp.execute("while True: pass")
    except CodeExecutionError as err:
        print("timed out:", err)
    interp.shutdown()
