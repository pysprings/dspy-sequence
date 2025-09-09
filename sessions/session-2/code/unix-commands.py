import dspy
from dspy.teleprompt import MIPROv2


# Step 1: Define the task signature
class EnglishToUnix(dspy.Signature):
    """Convert natural language requests into Unix commands."""

    request = dspy.InputField(desc="what the user wants to do")
    command = dspy.OutputField(desc="the Unix command to accomplish this")


# Step 2: Create the DSPy module
class UnixCommandGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        # Using ChainOfThought for better reasoning about command construction
        self.generate = dspy.ChainOfThought(EnglishToUnix)

    def forward(self, request):
        return self.generate(request=request)


# Step 3: Create training examples
# These define the input-output space we want to learn
training_examples = [
    dspy.Example(
        request="list all files in the current directory", command="ls -la"
    ).with_inputs("request"),
    dspy.Example(request="show me what's in this folder", command="ls").with_inputs(
        "request"
    ),
    dspy.Example(
        request="find all Python files in the current directory and subdirectories",
        command="find . -name '*.py'",
    ).with_inputs("request"),
    dspy.Example(
        request="count the number of lines in file.txt", command="wc -l file.txt"
    ).with_inputs("request"),
    dspy.Example(
        request="show the last 20 lines of the log file", command="tail -n 20 log.txt"
    ).with_inputs("request"),
    dspy.Example(
        request="search for the word 'error' in all log files",
        command="grep 'error' *.log",
    ).with_inputs("request"),
    dspy.Example(
        request="make a new directory called projects", command="mkdir projects"
    ).with_inputs("request"),
    dspy.Example(
        request="delete the file called temp.txt", command="rm temp.txt"
    ).with_inputs("request"),
    dspy.Example(request="show running processes", command="ps aux").with_inputs(
        "request"
    ),
    dspy.Example(
        request="see how much disk space is available", command="df -h"
    ).with_inputs("request"),
    dspy.Example(
        request="compress the logs folder into an archive",
        command="tar -czf logs.tar.gz logs/",
    ).with_inputs("request"),
    dspy.Example(
        request="show me the current directory path", command="pwd"
    ).with_inputs("request"),
    dspy.Example(
        request="copy file.txt to backup.txt", command="cp file.txt backup.txt"
    ).with_inputs("request"),
    dspy.Example(
        request="move document.pdf to the Documents folder",
        command="mv document.pdf Documents/",
    ).with_inputs("request"),
    dspy.Example(
        request="show the contents of config.json", command="cat config.json"
    ).with_inputs("request"),
    dspy.Example(
        request="find files larger than 100MB", command="find . -size +100M"
    ).with_inputs("request"),
    dspy.Example(
        request="check network connectivity to google.com", command="ping google.com"
    ).with_inputs("request"),
    dspy.Example(request="show system information", command="uname -a").with_inputs(
        "request"
    ),
    dspy.Example(
        request="list all running Docker containers", command="docker ps"
    ).with_inputs("request"),
    dspy.Example(
        request="see who is logged into the system", command="who"
    ).with_inputs("request"),
]


# Step 4: Define a simple exact match metric
def command_match_metric(example, prediction, trace=None):
    """Checks if the predicted command exactly matches the gold command."""
    return example.command.strip() == prediction.command.strip()


# Step 5: Configure DSPy with a language model
# Replace with your preferred model
lm = dspy.LM("openai/gpt-3.5-turbo", max_tokens=200)
dspy.settings.configure(lm=lm)

# Step 6: Initialize the base module
unix_generator = UnixCommandGenerator()

# Step 7: Optimize the module using MIPROv2 in zero-shot mode
optimizer = MIPROv2(
    metric=command_match_metric,
    auto="light",  # Can choose between light, medium, and heavy optimization runs
)

# Compile the optimized module
print("Optimizing the Unix command generator with MIPROv2 (zero-shot)...")
optimized_generator = optimizer.compile(
    unix_generator,
    trainset=training_examples,
    max_bootstrapped_demos=0,  # No generated examples
    max_labeled_demos=0,  # No examples from training set
)

# Step 8: Use the optimized module
test_requests = [
    "show me all JavaScript files",
    "how much RAM is being used",
    "create a backup of the entire home directory",
    "find all files modified today",
    "show the first 10 lines of data.csv",
]

print("\n" + "=" * 50)
print("Testing the optimized Unix command generator:")
print("=" * 50 + "\n")

for request in test_requests:
    result = optimized_generator(request=request)
    print(f"Request: {request}")
    print(f"Command: {result.command}")
    print("-" * 40)

# With zero-shot optimization, there are no demos to inspect.
# The `dspy.inspect_history()` call below will show the optimized prompt.

# Display the final prompt sent to the LM for the last test request
print("\n" + "=" * 50)
print("Final optimized prompt:")
print("=" * 50)
dspy.inspect_history(n=1)
