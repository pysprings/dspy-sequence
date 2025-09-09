# DSPy Training Examples
## A Conceptual Guide

---

## The Paradigm Shift

### From Prompt Engineering to Teaching by Example

• **Traditional approach:** Write detailed prompts, tweak parameters, hope for the best
• **DSPy approach:** Show examples of what you want, let optimization handle the rest
• **Core insight:** It's easier to show than to tell

### Why Examples Matter

• Examples ARE your specification
• They define success without complex rules
• They capture nuance that's hard to articulate
• They make expertise accessible to automation

---

## What is a Training Example?

### The Fundamental Unit

• **Definition:** A concrete demonstration of desired input-output behavior
• **Purpose:** Teaches the system patterns through demonstration
• **Power:** Transforms implicit knowledge into system behavior

### The DSPy.Example Class

```python
# Simple and flexible
example = dspy.Example(
    question="What is the capital of France?",
    answer="Paris"
).with_inputs("question")

# Any fields, any structure
complex_example = dspy.Example(
    query="...",
    context="...",
    reasoning="...",
    answer="...",
    confidence=0.95
).with_inputs("query", "context")
```

### Key Properties

• **Flexible:** Any fields, any types
• **Immutable:** Operations return copies
• **Intuitive:** Dot notation or dictionary access
• **Explicit:** Clear separation of inputs vs outputs

---

## Anatomy of Good Examples

### The Four Pillars

**1. Representativeness**
• Cover real-world distribution
• Include messy, actual user inputs
• Don't just show ideal cases

**2. Diversity**
• Span different aspects of your problem
• Cover edge cases and common cases
• Avoid redundancy

**3. Consistency**
• Uniform structure across examples
• Consistent labeling conventions
• Clear patterns for the optimizer to find

**4. Quality**
• Each example is a vote for behavior
• Bad examples actively harm performance
• Quality > Quantity (but you need both)

---

## From Data to Examples

### Common Data Sources

**Existing Assets:**
• Customer support logs
• Historical interactions
• Documentation and FAQs
• Domain expert demonstrations

### Transformation Decisions

**Key Questions to Answer:**
• What exactly is the input?
• What constitutes success?
• What context is needed?
• What output format serves users best?

### Hidden Design Choices

• Need citations? → Add a "sources" field
• User expertise varies? → Add "user_level" input
• Want reasoning? → Include "reasoning" output
• **Your example structure defines system capabilities**

---

## Examples as Communication

### What Examples Teach

• **Explicitly:** Input-output mappings
• **Implicitly:** Style, tone, and priorities
• **Systematically:** Trade-offs and constraints

### Mixed Signals to Avoid

• Contradictory examples confuse optimization
• Inconsistent format suggests flexibility that doesn't exist
• Varying quality implies some examples don't matter

### Clear Signals to Send

• Consistent patterns enable strong optimization
• Uniform quality sets clear standards
• Thoughtful coverage guides generalization

---

## Building Your First Training Set

### Start Small and Focused

```python
# Minimal but complete training set
training_examples = [
    dspy.Example(
        question="What is the capital of Japan?",
        answer="Tokyo"
    ).with_inputs("question"),
    
    dspy.Example(
        question="Who painted the Mona Lisa?",
        answer="Leonardo da Vinci"
    ).with_inputs("question"),
    
    # 3-5 more examples covering different domains...
]
```

### What This Teaches

• **Style:** Concise, direct answers
• **Format:** No preamble or elaboration
• **Coverage:** Different knowledge domains
• **Consistency:** Uniform structure throughout

---

## Patterns for Success

### Do's

✅ **Progressive Complexity**
• Start simple, add nuance gradually
• Master basics before edge cases

✅ **Explicit Negative Examples**
• Show what NOT to do
• Include corrections

✅ **Track Provenance**
• Know where examples came from
• Understand your data's biases

✅ **Version Control**
• Track changes over time
• Document why changes were made

### Don'ts

❌ **Kitchen Sink Examples**
• Don't teach multiple concepts at once
• Keep examples focused

❌ **Retrofitting Bad Outputs**
• Don't perpetuate existing problems
• Critically evaluate before reusing

❌ **Ignoring Coverage Gaps**
• Identify what's missing
• Actively seek diverse examples

---

## Scaling Your Training Data

### Organization Strategies

**Categorization:**
• Group by difficulty level
• Separate by domain or use case
• Tag with metadata

**Maintenance:**
• Regular audits for quality
• Remove outdated examples
• Add new edge cases from production

**Documentation:**
• Record creation decisions
• Explain non-obvious choices
• Note assumptions and limitations

---

## The Philosophy

### Teaching vs Programming

**Traditional Programming:**
• Write explicit rules
• Handle every case manually
• Debug by adjusting logic

**Teaching Through Examples:**
• Demonstrate desired behavior
• Let patterns emerge
• Debug by improving examples

### Democratization of Development

• **Domain experts** can contribute without coding
• **Quality bar** is recognizing good outputs
• **Maintenance** becomes updating examples
• **Testing** is inherent in the training set

---

## Key Insights

### Examples Are Everything

• They're your specification
• They're your test suite
• They're your documentation
• They're your quality control

### Investment Returns

• **Time spent on examples = System quality**
• **Better examples > More optimization**
• **Clear examples = Predictable behavior**

### The Mental Model

• Think of examples as **teaching a smart colleague**
• Show what good looks like
• Be consistent in your expectations
• Cover the important cases

---

## Looking Forward

### Evolution of Examples

**Near Future:**
• Weighted examples for importance
• Automatic gap detection
• Interactive refinement tools

**Longer Term:**
• Context-aware example selection
• Community example repositories
• Example synthesis from requirements

### The Unchanging Truth

No matter how DSPy evolves:
• **Examples remain fundamental**
• **Quality determines success**
• **Consistency enables optimization**
• **Coverage defines capabilities**

---

## Summary: The Example-First Mindset

### Remember

• **Examples > Instructions**
• **Showing > Telling**
• **Demonstration > Description**

### Your Examples Are

• How you teach the system
• What the system learns
• Why the system behaves as it does

### Success Comes From

• Thoughtful example selection
• Consistent patterns
• Comprehensive coverage
• Continuous refinement

---

## One Final Thought

**"In DSPy, you don't write programs—you curate examples that become programs."**

Your training examples are the DNA of your system. Invest in them accordingly.

---

*Based on consolidated DSPy training examples conversations*
