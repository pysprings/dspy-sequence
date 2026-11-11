# Session 9: Assertions - Reliable AI Systems
*DSPy Mastery Series - Month 9*

<style>pre { font-size: 0.9em; }</style>

## 1. Opening: Check the Reply Before the Program Sends It

### Where We Are

![The full DSPy map: a Foundation band (LM, Examples, Signatures), a Core band (Adapters, Modules, Metrics), an Advanced band (Optimizers, Advanced Modules, Assertions), and a Production band (Trackers), connected top to bottom.](assets/images/dspy-architecture.png)

We are at the last box of the Advanced band.

Session 6 gave us a metric. It scores a program on a set of examples.
Session 7 gave the metric to an optimizer, and the optimizer improved
the prompts. Session 8 changed the program by hand and gave it a
knowledge base to read.

All three sessions used the metric after the program had finished its
work. Today the program checks its own output while it works. When
the output fails the check, the program tries again.

### The system so far

Planbook is a fictional project management product. Its customers
send email to a support address. Our program reads one such email and
writes the first reply. Session 8 named the program the **grounded
inbox**, because it reads a knowledge base before it writes.

![Overview of the grounded inbox. An email goes to triage, which labels it. For bug reports only, extract writes a ticket. Search finds three passages in a knowledge base of 32 passages. Draft writes the reply from the email, the labels, and the passages.](assets/images/grounded-inbox-overview.svg)

The program has four steps:

1. `triage` labels the email with a category (billing, bug report,
   or feedback) and an urgency (low or high).
2. `extract` runs on bug reports only. It writes a one-line ticket
   for the engineers.
3. `search` finds the three knowledge base passages most like the
   email.
4. `draft` receives the email, the category, and the three passages.
   The passages arrive in a field named `context`. It writes a reply
   of two or three sentences.

Plain Python then adds `[PRIORITY]` to the reply when the urgency is
high.

The next diagram shows the same program in more detail. Each arrow
carries a named piece of data.

![Data flow of the grounded inbox. The email goes to three steps: triage, extract, and search. Triage is a predictor and produces a category and an urgency. Extract is a predictor that runs on bug reports only and produces a ticket. Search is plain Python: a retriever that reads a knowledge base of 32 passages and returns three passages as context. The draft predictor receives the email, the category, and the context, and writes a reply. Plain Python then adds a PRIORITY prefix when the urgency is high. The result is the reply to the customer.](assets/images/grounded-inbox-flow.svg)

The diagram has two kinds of step. A **predictor** is a module that
calls the language model with a prompt. `triage`, `extract`, and
`draft` are predictors. `search` is a retriever, which is plain
Python. An optimizer can change only predictors, so no optimizer can
change `search`.

### One email, step by step

Follow one test email through the program:

> I was charged twice for my March invoice. Please refund the
> duplicate.

1. `triage` returns the category `billing` and the urgency `high`.
2. `extract` does not run, because the email is not a bug report.
3. `search` returns three passages. This is the first:

   > Report a suspected duplicate charge to support@planbook.example
   > and include both charge dates, amounts, and the last four digits
   > of the card. Duplicate-charge reports are high urgency because
   > money has already been taken, and billing support responds
   > within two business days.

   The other two passages cover refunds and invoice disputes. Each of
   them also tells the reader to email support.
4. `draft` writes a reply from the email, the category, and the
   passages.
5. The urgency is high, so the program adds `[PRIORITY]`.

The customer receives this reply:

> [PRIORITY] Thank you for bringing this to our attention. Please
> email support@planbook.example with the charge dates, amounts, and
> the last four digits of your card so we can promptly review and
> assist with the refund.

### How we score a reply

The earlier sessions built these pieces around the program:

| Piece | What it is | Session |
|---|---|---|
| data | 36 support emails: 24 to train, 6 to validate, 6 to test | 7 |
| metric | `inbox_metric`: four checks for each email | 6 |
| task model | `gpt-4.1-nano`, a small model that writes every label and every reply | 5 |
| larger model | `gpt-4.1-mini`, which costs four times as much for each token | 6 |
| judge | a predictor on the larger model that answers two yes/no questions about a reply | 6 |
| knowledge base | 32 passages about Planbook's rules and known problems | 8 |

The metric has four checks for each email. Two compare a label with
the correct answer, which a person wrote for each email:

- `category`: is the category correct?
- `urgency`: is the urgency correct?

Two ask the judge a question about the reply. We call these two the
**judged checks**:

- `grounded`: "Does the reply address this customer's specific
  situation rather than being a generic acknowledgement?"
- `actionable`: "Does the reply state a concrete next step or
  resolution the customer can expect?"

The `grounded` check shares a word with the program's name, but it is
a different idea. The check asks if the reply is specific to this
customer. The name says that the program reads a knowledge base.

For the email above, both labels are correct. The judge answers no to
`grounded` and yes to `actionable`. So this email passes three of its
four checks.

The test score is the percentage of checks that pass on the six test
emails. Session 8 ended at 75.0, which is 18 of 24 checks. Every
label was correct. Half of the judged checks failed.

### What is wrong with it

Read the reply above as the customer does. They wrote to support
about a double charge. Support answers: write to support about your
double charge. The reply solves nothing. The customer must send a
second email and wait again.

A reply that tells the customer to email support is a **redirect**.
Here are the six test replies from Session 8. The last column is new:

| Test email | grounded | actionable | redirect |
|---|---|---|---|
| export button crashes, we ship today | no | yes | **yes** |
| card declined, invoice unpaid twice | yes | no | **yes** |
| charged twice, refund the duplicate | no | yes | **yes** |
| settings page misspells preferences | yes | yes | no |
| add a CSV export option someday | yes | no | no |
| new dashboard layout is a big improvement | no | no | no |

Three of the six replies are redirects.

**Why the program writes redirects.** Session 8 found the cause. The
passages are help center pages. A help center page is written for a
customer who has not yet contacted support, so it says "email
support." The instructions of `draft` say to use the passages. The
model did so, and copied the sentence into the reply.

**Why the score does not show them.** No metric check looks for a
redirect. Two of the three redirects even pass `actionable`, because
"email us these details" is a concrete next step. A reply can pass a
check and still be a reply that we do not want to send.

The table shows three problems. They are of different kinds.

1. **Three of six replies are redirects.** This fault has one cause
   and a simple test: the reply contains the support address. One
   line of Python can find it.
2. **Half of the judged checks fail.** These faults have no single
   cause. One reply is too general. Another gives no next step. Only
   the judge can find them.
3. **The program sends every reply anyway.** The first two problems
   reach the customer because of this one.

### Why the program sends a bad reply

Look at what `forward` does after `draft` writes a reply. Here is a
sketch for the charged-twice email:

```text
forward(email)
  triage(email)                   -> billing, high
  search(email)                   -> 3 passages
  draft(email, category, context) -> "... Please email support@planbook.example ..."
  add "[PRIORITY]"                   the urgency is high
  return the reply                   no line has read the reply
```

The program has no way to tell a good reply from a bad one.

The metric can tell them apart, but it does not run in the program:

- **It runs later.** We run the metric at our desks, after the
  program has finished.
- **It runs on the test set.** The metric needs the correct labels,
  and only our 36 emails have them. A new email from a real customer
  has none.

So in production nothing checks a reply. A customer receives a
redirect, and neither the program nor we learn of it.

### Why the earlier tools do not solve this

The tools from Sessions 7 and 8 work before the program runs:

- An optimizer changes the prompt once. The same prompt then serves
  every email.
- A change to the knowledge base works the same way. The same
  passages then serve every email.

Both can lower the number of bad replies, and §8 uses one of them.
Neither can catch the bad replies that remain. Some will remain,
because each reply is a random sample from the model. The same prompt
gives a good reply to one email and a bad reply to the next.

```text
one prompt, two emails
  "... the settings page misspells 'preferences' ..." -> a reply that passes
  "I was charged twice for my March invoice. ..."     -> "... Please email support@..."
```

To catch a bad reply, something must read each reply after `draft`
writes it and before the program sends it. That check must work
without a correct label. This session adds it.

### What we add today

We put a loop around the `draft` step. The loop gives each reply a
score, called a **reward**, before the program sends it. When the
reward is low, the loop asks for another reply.

Here is a sketch of one such loop on the charged-twice email. §5
shows the full output:

```text
attempt 1  "... Please email support@planbook.example ..."    reward 0.0  try again
attempt 2  "... We are currently reviewing your account ..."  reward 0.5  try again
attempt 3  "... We have flagged your duplicate charge ..."    reward 1.0  send
```

Three more terms describe the loop:

- An **attempt** is one run of the wrapped step.
- The **threshold** is the reward at which the loop stops and sends
  the reply.
- A **hint** is a note that one of the two loops gives to the next
  attempt.

![The retry loop around the draft step. The email, category, and context go into draft. Draft makes one attempt and produces a reply. A reward function scores the reply with a hard rule and then a judge. If the reward is at the threshold, the reply is sent. If not, and attempts remain, the loop returns to draft. BestOfN drafts again from the same prompt. Refine first calls a hint writer, a predictor that Refine creates, and drafts again with the hint. When no attempt remains, both loops send the highest-scoring attempt.](assets/images/retry-loop.svg)

DSPy has two such loops. Both are modules that wrap another module.

| Addition | What it does | Where we cover it |
|---|---|---|
| a reward function | scores one reply from 0.0 to 1.0, with no correct answer to compare with | §3, The Reward Function: we write it and score the six Session 8 replies |
| `dspy.BestOfN` | runs the step up to N times and keeps the best reply | §4, BestOfN: Draft Again: we wrap `draft` and read every attempt |
| `dspy.Refine` | does the same, and gives each new attempt a hint about the last one | §5, Refine: Draft Again With a Hint: we read the hints and choose the model that writes them |
| a measurement | four programs, three runs each, on the six test emails | §6, The Measurement: scores, redirects, and calls per email |

**Session Goals:**

- Explain what replaced `dspy.Assert`
- Write a reward function, and say how it differs from a metric
- Put `BestOfN` around one step of a program and read every attempt
- Put `Refine` around the same step and read the hints
- Measure both loops against the program with no loop: score, and
  calls per email
- Name four things a retry loop does not do

## 2. From Assertions to Reward Functions

DSPy 2 called a check inside the program an assertion. This session
keeps the name, although DSPy 3 replaced the feature with the retry
loops of §1.

DSPy 2 had two assertion functions. You wrote them inside `forward`:

```python
# DSPy 2. Removed in DSPy 3. Do not copy.
dspy.Assert("support@" not in reply, "Do not tell the customer to email support.")
dspy.Suggest(len(reply) < 400, "Keep the reply short.")
```

Both worked the same way. When the condition was false, DSPy ran the
step again and put your message in the prompt. After two failures,
`Assert` raised an error and `Suggest` continued.

Here is a sketch of that retry for the charged-twice email from §1.
The `...` marks text that we left out:

```text
attempt 1
  reply  : "... Please email support@planbook.example with the charge dates ..."
  Assert : "support@" is in the reply, so the condition is false

attempt 2, the prompt gains two parts
  the failed reply : "... Please email support@planbook.example ..."
  your message     : "Do not tell the customer to email support."
  reply  : (a new reply)
```

DSPy 3 removed both. `dspy.BestOfN` and `dspy.Refine` replace them.
The program still checks an output and still tries again. Three
details changed:

| | DSPy 2 assertions | DSPy 3 retry loops |
|---|---|---|
| the check | a true/false condition | a **reward function** that returns a number |
| where it lives | a line inside `forward` | a module that wraps a step |
| the retry message | a sentence that you write | a hint that a model writes (`Refine`), or no message (`BestOfN`) |

### A reward function is not a metric

The two look alike. Both are plain Python functions that return a
score. The difference is what each one receives.

| | Metric (Session 6) | Reward function (today) |
|---|---|---|
| signature | `metric(example, pred, trace=None)` | `reward_fn(args, pred)` |
| first argument | a gold example, with the correct labels | the inputs of the wrapped step, as a dict |
| when it runs | after the program, on a data set | inside the program, on every call |
| who calls it | `dspy.Evaluate` or an optimizer | `BestOfN` or `Refine` |
| when you pay | once for each experiment | on every email |

Here is what each function receives for the charged-twice email:

```text
metric(example, pred)
  example.email    = "I was charged twice for my March invoice. ..."
  example.category = "billing"      <- written by a person: the correct label
  example.urgency  = "high"         <- written by a person: the correct label
  pred.reply       = "... Please email support@planbook.example ..."

reward_fn(args, pred)
  args["email"]    = "I was charged twice for my March invoice. ..."
  args["category"] = "billing"      <- written by triage: it can be wrong
  args["context"]  = ["Report a suspected duplicate charge to support@...", ..., ...]
  pred.reply       = "... Please email support@planbook.example ..."
```

Both receive the same reply. Only the metric receives a label that a
person wrote.

The first-argument row matters most. A reward function runs in
production, on an email that nobody has labeled. It has no correct
answer to compare with. That gives us the rule for today:

**A reward function can use only what the program has when it runs.**

Apply the rule to our four checks:

| Check | Needs a correct label? | Can it be a reward? |
|---|---|---|
| `category` | yes | no |
| `urgency` | yes | no |
| `grounded` | no, the judge reads the email and the reply | yes |
| `actionable` | no, the judge reads the email and the reply | yes |

Two of the four checks carry over. The retry loop cannot fix a wrong
label, because it cannot know that the label is wrong.

### Version notes

This session's code and numbers use DSPy 3.3.0. To run the scripts,
install that version:

```bash
pip install "dspy==3.3.0"
```

DSPy 3.5.0 came out on 10 November 2026, the day of this session.
What it means for today:

- **The two loops are the same.** `BestOfN` has no change. `Refine`
  has one fix: the hint writer now also receives the reasoning of a
  `ChainOfThought` step such as `draft`.
- **`Refine` still has no argument for the hint writer's model.** The
  remedy in §5 still applies.
- **Our scripts do not run on 3.5.0 without changes.** The first
  failure is the embedder: 3.5.0 rejects a model name that starts
  with `openrouter/`.
- **Every prompt changes.** The default adapter is now `XMLAdapter`,
  so a prompt has a different format. Expect scores that differ from
  the ones in this deck.
- **Session 8 is affected too.** 3.5.0 removes `dspy.CodeAct` and
  `dspy.ProgramOfThought`, and `dspy.ReActV2` is now `dspy.ReAct`.

## 3. The Reward Function

### What we do

We write one function, `reply_reward`, in two steps. Then we use it
to score the six replies that we already have. This section retries no reply. It only shows the reward for each one.

### Step 1: a hard rule in plain Python

Three replies are redirects. Every one of them contains the support
address. One line finds it:

```python
def redirects_customer(reply: str) -> bool:
    """True if the reply tells the customer to email support. They already did."""
    return "support@" in reply.lower()
```

```
--- the hard rule: one line of Python ---
redirects_customer = True  | Please email support@planbook.example with the charge dates.
redirects_customer = False | We have your report and will answer within two business days.
```

This check costs nothing and gives the same answer every time. Use a
check like this whenever one exists.

### Step 2: the two judged checks

The `grounded` and `actionable` checks need the judge. We take the
judge from Session 6 without a change, with the same two questions:

```python
def judge_reply(email: str, reply: str) -> dict[str, bool]:
    """The metric's two judged checks. Neither needs a gold label."""
    with dspy.context(lm=JUDGE_LM):
        grounded = assess(email=email, reply=reply, question=GROUNDED_Q).verdict
        actionable = assess(email=email, reply=reply, question=ACTIONABLE_Q).verdict
    return {"grounded": grounded, "actionable": actionable}
```

`assess` is the judge: a predictor that receives the email, the
reply, and one question, and returns a true/false `verdict`.
`dspy.context` selects the larger model for those two calls.
`JUDGE_LM` is the larger model and `TASK_LM` is the task model, both
from the table in §1.

### The whole function

```python
def reply_reward(args, pred) -> float:
    """Score one draft reply from 0.0 to 1.0.

    A hard rule comes first. If the reply tells the customer to email
    support@..., the reward is 0.0. The customer is already writing to
    support, so the reply must handle the request itself.

    Two judged checks follow, each worth half a point:
    grounded:   the reply addresses this customer's specific situation and
                is not a generic acknowledgement.
    actionable: the reply states a concrete next step or resolution that
                the customer can expect.
    """
    if redirects_customer(pred.reply):
        return 0.0
    checks = judge_reply(args["email"], pred.reply)
    return (checks["grounded"] + checks["actionable"]) / 2
```

Follow the function for the reply to the charged-twice email:

```text
reply_reward(args, pred)
  pred.reply = "... Please email support@planbook.example with the charge dates ..."
  redirects_customer(pred.reply)   ->  True
  return 0.0                           the judge does not run
```

A reply with no redirect takes the longer path. This sketch uses a
reply that passes one judged check:

```text
reply_reward(args, pred)
  pred.reply = "... We are currently reviewing your account details ..."
  redirects_customer(pred.reply)   ->  False
  judge_reply(email, reply)        ->  grounded False, actionable True
  return (0 + 1) / 2               ->  0.5
```

Three decisions are in this function:

1. **The hard rule overrides the other checks.** A reply that breaks it scores 0.0,
   whatever else it does well. The rule does the job of `Assert`, but it raises no error. The two
   judged checks share one point. They do the job of `Suggest`.
2. **The hard rule runs first, because it is free.** When the rule
   fails, the function
   returns before it calls the judge. We do not pay a judge to score
   a reply that we already reject.
3. **The docstring says what each check means.** §5 needs this
   docstring, because `Refine` puts this function's source code in a
   prompt.

### The output

`code/reward-function.py` runs the grounded inbox on the six test
emails and scores each reply:

```
--- the six test replies from Session 8, scored by the reward ---
hard rule  grounded  actionable  reward   email
     FAIL         -           -    0.00   URGENT: the export button crashes the app
     FAIL         -           -    0.00   My card was declined but you still marked
     FAIL         -           -    0.00   I was charged twice for my March invoice.
     pass      pass        pass    1.00   Minor thing: the settings page misspells '
     pass      pass        FAIL    0.50   Could you add a CSV export option someday?
     pass      FAIL        FAIL    0.00   Just wanted to say the new dashboard layou

replies at reward 1.0 : 1 of 6
mean reward           : 0.25
```

### What it shows

One reply in six passes. With a threshold of 1.0, a retry loop has
five replies to work on.

A dash means that the judge did not run. The hard rule rejected three
replies first.

The last row needs an explanation. It is the praise email:

> Just wanted to say the new dashboard layout is a big improvement.
> Nice work.

This customer reports no problem and asks for nothing. The program
wrote a thank-you:

> Thank you for your kind words! We're glad to hear that you like
> the new dashboard layout and appreciate your feedback. If you have
> any suggestions or run into any issues, feel free to reach out.

For a person, this reply is fine. The judge fails it on both checks.
We asked the judge to give its reasons:

- `actionable`: the reply "does not state a concrete next step or a
  resolution that the customer can expect." That is true. The
  customer needs nothing, so no next step exists.
- `grounded`: the reply is "a general thank-you message followed by
  a standard invitation for further contact." That is also true. The
  email gives one detail, the dashboard layout, and the reply repeats
  it. The email holds nothing more to be specific about.

So the judge answers both questions correctly, and a good reply
scores 0.0. The fault is in the reward, not in the reply and not in
the judge. We wrote both questions for a customer who has a problem.
This customer has none.

We leave this fault in the reward on purpose. It shows what a retry
loop does with an email that cannot pass. In §4 and §5 the loop uses
all three attempts on this email and still fails. §7 (What the Loops
Do Not Do) then stops the program from sending such a reply without a
person's review.

The repair of the reward itself is an exercise for you, in the action
items at the end of the session. One way is to skip the `actionable`
check when the category is `feedback`.

## 4. BestOfN: Draft Again

### What we do

We wrap the `draft` step in `dspy.BestOfN`. When a reply scores below
the threshold, the loop asks the model for another reply to the same
prompt. It stops when a reply reaches the threshold, or after three
attempts.

### The code

```python
inbox = GroundedInbox()
inbox.draft = dspy.BestOfN(module=inbox.draft, N=3, reward_fn=reply_reward, threshold=1.0)
```

`forward` needs no change. It still calls `self.draft(...)`, and the
step now runs up to three times.

### How it works

`BestOfN` is about 30 lines of Python. For each attempt it does this:

1. It copies the wrapped module and gives the copy a new
   `rollout_id`. DSPy stores each model call in a cache, so the same
   prompt returns the stored output. A new `rollout_id` skips the
   stored output. The model then produces a new output, which we call
   a **sample**.
2. It sets the temperature to 1.0. Temperature controls how random
   the output is. At 1.0, two samples from one prompt differ.
3. It runs the copy and calls `reward_fn(args, pred)`.
4. If the reward is at the threshold or above, it returns that
   prediction.

After N attempts with no pass, it returns the attempt with the
highest reward. When two attempts tie, the earlier one wins.

Here is a sketch of those steps for the typo email. The output below
shows the replies:

```text
attempt 1  rollout_id 0, temperature 1.0 -> sample 1 -> reply_reward -> 0.5
           0.5 is below the threshold of 1.0, so the loop goes on
attempt 2  rollout_id 1, temperature 1.0 -> sample 2 -> reply_reward -> 1.0
           1.0 is at the threshold, so the loop returns sample 2
attempt 3  does not run
```

`named_predictors()` lists the predictors that an optimizer can
change. The wrapped predictor is still in the list, one level deeper.
So a Session 7 optimizer can still change it when the loop is in
place:

```
--- the predictor tree, with and without the loop ---
GroundedInbox : ['triage', 'extract', 'draft.predict']
with BestOfN  : ['triage', 'extract', 'draft.module.predict']
```

### The output

The loop returns only the final prediction. To show every attempt,
`code/best-of-n.py` wraps `reply_reward` in a recorder. First, an
email where the loop works:

```
--- an email where a second sample is enough ---
email : Minor thing: the settings page misspells 'preferences' in the sidebar.
  attempt 1  reward 0.50  Thank you for pointing out the typo in the settings
                          sidebar. We appreciate your feedback and will review and
                          address it in a future update.
  attempt 2  reward 1.00  Thank you for bringing this to our attention. We will pass
                          your note about the misspelling of 'preferences' in the
                          settings sidebar to our development team for review.
  sent  : Thank you for bringing this to our attention. We will pass your note about
          the misspelling of 'preferences' in the settings sidebar to our
          development team for review.
```

The judge passed one check on the first reply and both checks on the
second, so the loop stopped. Attempt 1 is a new sample, so it is not
the reply from §3. Now an email where the loop does not work:

```
--- an email where three samples are not enough ---
email : I was charged twice for my March invoice. Please refund the duplicate.
  attempt 1  reward 0.00  Thank you for bringing this to our attention. Please email
                          support@planbook.example with your workspace name, invoice
                          number, charge date, and the last four digits of your
                          card, so we can investigate the duplicate charge promptly.
  attempt 2  reward 0.00  Thank you for reaching out. Please email
                          support@planbook.example with the charge date, invoice
                          number, and the last four digits of the card to report
                          this duplicate charge, and our team will respond within
                          two business days.
  attempt 3  reward 0.00  Thank you for bringing this to our attention. Please email
                          us at support@planbook.example with the details of both
                          charges, including the charge dates, amounts, and the last
                          four digits of your card, so we can investigate and
                          process a refund if applicable. We will respond within two
                          business days.
  sent  : [PRIORITY] Thank you for bringing this to our attention. Please email
          support@planbook.example with your workspace name, invoice number, charge
          date, and the last four digits of your card, so we can investigate the
          duplicate charge promptly.
```

All three samples are redirects.

### The test result

"One draft" is the grounded inbox with no loop.

```
--- test split: one draft against BestOfN ---
program                    score  category  urgency  grounded  actionable  redirects
one draft                   75.0       6/6      6/6       3/6         3/6        3/6
BestOfN, N=3                79.2       6/6      6/6       2/6         5/6        3/6
attempts per email : one draft 1.0   BestOfN 2.8

--- the attempts, per email ---
attempt 1  attempt 2  attempt 3 | email
     0.00       0.00       0.00 | URGENT: the export button crashes the app for ou
     0.00       0.00       0.00 | My card was declined but you still marked the in
     0.00       0.00       0.00 | I was charged twice for my March invoice. Please
     0.50       1.00            | Minor thing: the settings page misspells 'prefer
     0.50       0.00       0.50 | Could you add a CSV export option someday? Not b
     0.00       0.50       0.00 | Just wanted to say the new dashboard layout is a
```

A blank cell means that the loop had stopped. In the first three
rows, each 0.00 is a redirect.

### What it shows

**A new sample can fix a fault that chance causes. It does not fix a
fault that the prompt causes.**

The typo email got a weak first reply by chance. A second sample was
better. That is the case `BestOfN` is made for.

The three emails with redirects are a different case. All nine attempts are redirects. The cause is in the prompt.
For these emails, every passage that the draft step receives contains
the address support@planbook.example. We checked all nine passages.
Here is a sketch of the prompt for the charged-twice email. It is the
same for all three attempts:

```text
instructions : Write a short, courteous reply ... Use the context passages ...
email        : I was charged twice for my March invoice. Please refund the duplicate.
category     : billing
context      : [1] Report a suspected duplicate charge to support@planbook.example ...
               [2] ... Email support@planbook.example with the workspace name ...
               [3] Dispute an invoice by emailing support@planbook.example with ...

attempt 1 -> "... Please email support@planbook.example with your workspace name ..."
attempt 2 -> "... Please email support@planbook.example with the charge date ..."
attempt 3 -> "... Please email us at support@planbook.example with the details ..."
```

The instructions say to use the passages, and all three passages say
to email support. Every sample comes from that prompt, so every
sample repeats the fault. More samples add cost and do not remove it.

The loop used 2.8 attempts for each email. `actionable` gained two
checks and `grounded` lost one, so the net change is one check. §6 shows that chance alone moves the score by more than one check.

The loop sent this reply to the customer who praised the dashboard:

> Thank you for your kind words about the new dashboard layout! We're
> aware of the issue with custom date ranges after a workspace
> time-zone change and are working to resolve it within the next
> week. Meanwhile, reloading the page and re-selecting the date range
> can serve as a temporary workaround.

The customer reported no problem. The reply describes a bug from the
knowledge base and offers a workaround. It scored 0.5, the best of
three, because the judge found a concrete next step in it. The loop kept the reply with the highest reward. §7 explains why the
reward allows this.

## 5. Refine: Draft Again With a Hint

### What we do

We replace `BestOfN` with `dspy.Refine`. The loop is the same, with
one new step. After a failed attempt, a predictor that `Refine`
creates reads what happened and writes a hint. The next attempt receives the hint.

### The code

```python
inbox = GroundedInbox()
inbox.draft = dspy.Refine(module=inbox.draft, N=3, reward_fn=reply_reward, threshold=1.0)
```

Only the class name differs from §4.

### How the hint is made

We call this predictor the **hint writer**. It is hidden:
`named_predictors()` does not list it, as §7 shows. After each attempt
below the threshold, `Refine` calls it with these inputs:

```
--- what the hint writer reads ---
  program_code        The code of the program that we are analyzing
  modules_defn        The definition of each module in the program, including its I/O
  program_inputs      The inputs to the program that we are analyzing
  program_trajectory  The trajectory of the program's execution, showing each module's I/O
  program_outputs     The outputs of the program that we are analyzing
  reward_code         The code of the reward function that we are analyzing
  target_threshold    The target threshold for the reward function
  reward_value        The reward value assigned to the program's outputs
  module_names        The names of the modules in the program, for which we seek advice
```

`reward_code` holds the source code of your reward function. `Refine`
reads it with Python's `inspect` module and puts the text in a
prompt. The hint writer receives the reward value and the code that
produced it. It must work out from the code why the reply failed.

This is the reason for the docstring in §3. The names and comments
in `reply_reward` are the only explanation of the checks that the
hint writer gets. A reward function with the body
`return score(args, pred)` tells it nothing.

Here is a sketch of four of those inputs after attempt 1 on the
charged-twice email:

```text
program_trajectory : email   "I was charged twice for my March invoice. ..."
                     context ["Report a suspected duplicate charge to support@...", ...]
                     reply   "... Please email support@planbook.example with ..."
reward_code        : def reply_reward(args, pred) -> float:
                         """... If the reply tells the customer to email
                         support@..., the reward is 0.0. The customer is already
                         writing to support, ..."""
                         if redirects_customer(pred.reply):
                             return 0.0
                         ...
reward_value       : 0.0
target_threshold   : 1.0
```

The hint writer can match the reward of 0.0 to the first `return` in
the code. The docstring above that line says why the reply is a
fault.

The hint reaches the next attempt as one more input field:

```
--- the prompt of a hinted attempt gains one input field ---
1. `email` (str):
2. `category` (str):
3. `context` (list[str]): passages from the help center
4. `hint_` (str): A hint to the module from an earlier run
1. `reasoning` (str):
2. `reply` (str): 2-3 sentences
```

The first four lines are the input fields. The last two are the
output fields.

For the charged-twice email, attempt 2 receives these inputs. The
hint comes from the larger model, later in this section:

```text
email    : I was charged twice for my March invoice. Please refund the duplicate.
category : billing
context  : ["Report a suspected duplicate charge to support@...", ..., ...]
hint_    : The predict module made a mistake by redirecting the customer to email
           support again ... it should produce a reply that directly addresses the
           concern ... without asking for redundant action. ...
```

The passages still say to email support. The hint is the only new
text in the prompt.

### The first run

`code/refine.py` runs the email that `BestOfN` did not fix:

```
--- Refine, with the task model writing the hints ---
email : I was charged twice for my March invoice. Please refund the duplicate.
  attempt 1  reward 0.00  Thank you for bringing this to our attention. Please email
                          support@planbook.example with your workspace name, invoice
                          number, charge date, and the last four digits of your
                          card, so we can investigate the duplicate charge promptly.
  hint from the task model: N/A
  attempt 2  reward 0.00  Thank you for reaching out. Please email
                          support@planbook.example with the charge dates, amounts,
                          and the last four digits of your card so we can promptly
                          investigate and issue a refund if a duplicate charge is
                          confirmed. We respond to such reports within two business
                          days.
  hint from the task model: N/A
  attempt 3  reward 0.00  Thank you for reaching out. Please email
                          support@planbook.example with both charge dates, amounts,
                          and the last four digits of the card to verify and process
                          the refund promptly.
  sent  : [PRIORITY] Thank you for bringing this to our attention. Please email
          support@planbook.example with your workspace name, invoice number, charge
          date, and the last four digits of your card, so we can investigate the
          duplicate charge promptly.
```

The hint is "N/A." The hint writer received a reply with a reward of
0.0 and reported no fault.

The DSPy source explains part of this. The hint writer's instructions
allow the answer "N/A" when no module is at fault. The hint writer
also has no model of its own. It uses the model that `dspy.configure`
names, and we configured the task model. So the task model reviewed
its own reply and reported no fault. A larger model is the usual
remedy.

### The remedy: the larger model writes the hints

`Refine` has no argument for the hint writer's model. Two lines give
the same effect:

```python
inbox.set_lm(TASK_LM)                  # pin every predictor in the program to the task model
with dspy.context(lm=HINT_LM):         # the hint writer has no pin, so it uses this model
    pred = inbox(email=email)
```

```text
caller                      model          why
triage, extract             task model     pinned by set_lm
draft, on each attempt      task model     pinned by set_lm
hint writer                 larger model   no pin, so it takes the dspy.context model
judge, inside the reward    larger model   judge_reply opens its own dspy.context
```

This arrangement is clumsy. Nothing names the hint writer's model. It
gets the larger model only because every other predictor has a pin.
If you add a predictor and forget its pin, that predictor also runs
on the larger model, and nothing warns you. `Refine` has no argument
for the hint writer's model, in DSPy 3.3 or in 3.5.

`set_lm` pins each predictor to one model, so those predictors
ignore the configured model. `HINT_LM` is the larger model, which
already serves as the judge. Session 7 did the same for GEPA, the
optimizer that rewrites instructions: a larger model read the
failures of the task model.

```
--- Refine, with the larger model writing the hints ---
email : I was charged twice for my March invoice. Please refund the duplicate.
  attempt 1  reward 0.00  Thank you for bringing this to our attention. Please email
                          support@planbook.example with your workspace name, invoice
                          number, charge date, and the last four digits of your
                          card, so we can investigate the duplicate charge promptly.
  hint from the larger model: The predict module made a mistake by redirecting the
                              customer to email support again despite the customer
                              having already contacted support via email. This
                              mistake arises when the context leads it to rely on
                              processes involving customer outreach rather than
                              directly addressing the specific duplicate charge
                              issue. In such cases where the customer explicitly
                              reports a billing issue already communicated by email,
                              the predict module should not redirect to contact
                              support again. Instead, it should produce a reply that
                              directly addresses the concern by either confirming
                              investigation steps being initiated, providing
                              relevant policy details to reassure the customer, or
                              offering concrete next steps without asking for
                              redundant action. This concrete, customer-centric
                              response will avoid the reward penalty for redirection
                              and better fulfill the customer’s expectations.
  attempt 2  reward 0.50  Thank you for reporting the duplicate charge. We are
                          currently reviewing your account details and will respond
                          within two business days. Please rest assured that we are
                          working to resolve this as quickly as possible.
  hint from the larger model: Prior mistake scenario: None in this case since the
                              reply did not redirect the customer to email support
                              again, which is the key prior failure mode. Future
                              advice: Continue avoiding any replies that ask the
                              customer to reach out to support@planbook.example
                              again if they are already contacting support. Instead,
                              always produce a direct, grounded response that
                              addresses the customer's stated issue. To strengthen
                              the response further and achieve full reward, include
                              more specific details from the customer's email or
                              from context to provide concrete next steps (e.g.,
                              confirming investigation or refund processing
                              timelines) rather than only a general timeline. This
                              concreteness can improve judged scores on both
                              grounding and actionability.
  attempt 3  reward 1.00  Thank you for bringing this to our attention. We have
                          flagged your duplicate charge as high priority and will
                          review your account shortly. You can expect to hear back
                          within two business days with an update on your refund.
  sent  : [PRIORITY] Thank you for bringing this to our attention. We have flagged
          your duplicate charge as high priority and will review your account
          shortly. You can expect to hear back within two business days with an
          update on your refund.
```

In the hint, "the predict module" is the predictor inside `draft`.
The first hint names the fault and its cause. The docstring of
`reply_reward` gave it the fault: the customer is already writing to
support. The hint writer added the cause itself: the context led the
model to redirect the customer.

### The test result

```
--- test split ---
program                    score  category  urgency  grounded  actionable  redirects
one draft                   75.0       6/6      6/6       3/6         3/6        3/6
Refine, task-model hints    75.0       6/6      6/6       3/6         3/6        3/6
Refine, larger-model hints  87.5       6/6      6/6       5/6         4/6        0/6
attempts per email : task-model hints 3.0   larger-model hints 2.5
hints per email    : task-model hints 2.0   larger-model hints 1.5
task-model hints that say N/A : 11 of 12

--- the attempts with larger-model hints, per email ---
attempt 1  attempt 2  attempt 3 | email
     0.00       1.00            | URGENT: the export button crashes the app for ou
     0.00       1.00            | My card was declined but you still marked the in
     0.00       0.50       1.00 | I was charged twice for my March invoice. Please
     0.50       1.00            | Minor thing: the settings page misspells 'prefer
     0.50       0.00       0.50 | Could you add a CSV export option someday? Not b
     0.00       0.00       0.00 | Just wanted to say the new dashboard layout is a
```

### What it shows

**A hint can remove a fault that the prompt causes. A new sample
cannot.**

With hints from the larger model, no reply is a redirect.
Four of the six emails reach the threshold, and three of them need
only one hint.

In this run, with hints from the task model, `Refine` scored the same
as one draft. It used every attempt on every email. It paid for 12
hints, and 11 of them said "N/A."

Two more facts about the hints:

- **Each hint replaces the one before it.** The hint writer sees only
  the latest attempt and the hint that the attempt received. It does
  not see older attempts or older hints.
- **The hints are not kept.** When the call returns, the hints are
  gone. The next email starts with no hint. Three test emails needed
  the same hint, and the hint writer wrote it three times.

```text
email 1   attempt 1 -> hint A -> attempt 2 (reads hint A)
                    -> hint B -> attempt 3 (reads hint B only)
email 2   attempt 1 (no hint) -> hint A, written again -> ...
```

§8 moves the repeated hint into the prompt.

## 6. The Measurement

### What we do

Sections 4 and 5 each showed one run. One run on six emails is weak
evidence, for two reasons from earlier sessions:

- Six emails with four checks each make 24 checks. One changed check
  moves the score by 4.2 points.
- Every reply is a random sample. The same program scores
  differently each time it runs.

So `code/compare.py` runs four programs three times each. A run is a
fresh set of samples. The script changes the `rollout_id` to get one.

| Program | What it is |
|---|---|
| one draft | the grounded inbox from Session 8, with no loop |
| BestOfN | up to three attempts, keep the best |
| Refine, task-model hints | up to three attempts, the task model writes the hints |
| Refine, larger-model hints | up to three attempts, the larger model writes the hints |

### The scores

```
--- score on the test split, three runs ---
program                      run 1  run 2  run 3   mean
one draft                     75.0   79.2   66.7   73.6
BestOfN                       79.2   83.3   83.3   81.9
Refine, task-model hints      75.0   79.2   83.3   79.2
Refine, larger-model hints    87.5   95.8   83.3   88.9
```

The first row is one program, run three times with no change. Its
score ranges from 66.7 to 79.2, a spread of 12.5 points. So two
single runs must differ by more than 12.5 points before we call one
program better.

Now compare the other rows with the first:

- `Refine` with larger-model hints is ahead of one draft. Its worst
  run (83.3) is above the best run of one draft (79.2).
- `BestOfN` and `Refine` with task-model hints are a little ahead of
  one draft, and we cannot tell the two apart. With "N/A" hints,
  `Refine` is `BestOfN` at a higher cost.
- Against `BestOfN`, the runs of `Refine` with larger-model hints
  overlap at 83.3. The scores do not separate these two programs. The
  `redirects` column in the next table does.

### The checks

```
--- the checks, summed over three runs (18 replies) ---
program                      category  urgency  grounded  actionable  redirects
one draft                       18/18    18/18      9/18        8/18       8/18
BestOfN                         18/18    18/18     10/18       13/18       7/18
Refine, task-model hints        18/18    18/18     10/18       11/18       7/18
Refine, larger-model hints      18/18    18/18     13/18       15/18       0/18
```

The `redirects` column shows the largest difference. Three programs
write 7 or 8 redirects in 18 replies. `Refine` with larger-model hints writes
none.

The label columns did not move. The loop wraps the `draft` step only,
so `triage` runs once and its labels stay as they were.

### What one email costs

```
--- what one email costs, mean of three runs ---
program                      attempts  hints  judge calls
one draft                         1.0    0.0          0.0
BestOfN                           2.7    0.0          2.6
Refine, task-model hints          2.8    1.8          3.0
Refine, larger-model hints        2.5    1.5          3.7
(one judge call is one question; the reward asks two for each attempt that is
 not a redirect; the calls of triage, extract, and the metric are the same for
 every program and are left out)
```

These are means of three runs. §4 and §5 showed run 1 only, so their
numbers differ a little.

One draft makes one call to the task model for the reply. `Refine`
with larger-model hints makes 2.5 calls to the task model and about 5
calls to the larger model: 1.5 for hints and 3.7 for the judge inside
the reward. The larger model costs four times as much for each token. A
hint call is also long, because it carries source code and the last
attempt.

In dollars the amounts are small. `compare.py` makes 72 replies and
costs about 15 cents with an empty cache. All six scripts of this
session cost about 30 cents.

A metric costs money when you run an experiment. A reward function
costs this amount on every email.

### A second judge

The reward and the metric use the same judge. The loop selects
replies that this judge passes, so the metric can overstate the gain. Session 7
checked its judge against marks made by hand. Here we use a second
judge from another vendor, `claude-haiku-4.5`, with the same two
questions.

```
--- run 1's final replies, scored by a second judge ---
program                        first judge   second judge   (of 12 judged checks)
one draft                             6/12           5/12
BestOfN                               7/12           6/12
Refine, task-model hints              6/12           5/12
Refine, larger-model hints            9/12           7/12
```

The order of the programs is the same. The second judge also scores
`Refine` with larger-model hints highest, by two checks over one
draft. The first judge gives three. This is one run, and the first
table showed that chance moves a run by as much. So the second judge agrees
with the first, and it does not add proof.

## 7. What the Loops Do Not Do

§6 measured the gain. This section lists four limits of the loops.
`code/limits.py` shows limits 1, 3, and 4. Limit 2 comes from the
replies, which we read by hand. Every example is `Refine` with larger-model hints, run 1.

### Limit 1: a loop returns a reply even when every attempt fails

```
--- the loop returns a reply either way ---
email         : Just wanted to say the new dashboard layout is a big improvement.
                Nice work.
rewards tried : [0.0, 0.0, 0.0]
returned      : Thank you for the kind words! We're glad to hear that you find the
                new dashboard layout an improvement. We appreciate your support and
                will continue working to enhance your experience.
fields        : ['category', 'context', 'reply', 'ticket', 'urgency']
```

Three attempts scored 0.0, and the loop returned the first one. The
`rewards tried` line comes from our recorder. The prediction itself
has no `reward` field and no flag. The caller cannot tell
this reply from one that passed.

```text
a reply that passed    pred.reply = "... We have flagged your duplicate charge ..."
a reply that failed    pred.reply = "Thank you for the kind words! ..."
                       the two predictions have the same fields
```

`dspy.Assert` raised an error in this case. The loops do not. If a
reply below the threshold must not go out, write the check yourself:

```python
def send_or_hold(email: str, pred, threshold: float = 1.0) -> str:
    """Score the reply we are about to send. Below the threshold, a person reads it first."""
    return "send" if reply_reward({"email": email}, pred) >= threshold else "hold"
```

```
--- the same six emails, with a check before sending ---
send  | URGENT: the export button crashes the app for our whole team, we s
send  | My card was declined but you still marked the invoice unpaid twice
send  | I was charged twice for my March invoice. Please refund the duplic
send  | Minor thing: the settings page misspells 'preferences' in the side
hold  | Could you add a CSV export option someday? Not blocking, just hand
hold  | Just wanted to say the new dashboard layout is a big improvement.
sent 4 of 6, held 2 for a person
```

The script holds the praise email, but the reply is fine. The reward
is wrong for this email: it demands a next step where none exists. A reward function
has faults, as a metric does, and those faults now act on every
email.

### Limit 2: a loop keeps the reply with the highest reward, and the reward does not check truth

Here are two replies that `Refine` sent with a reward of 1.0:

> Thank you for pointing out the misspelling of "preferences" on the
> settings page. We will prioritize correcting this typo in the
> upcoming patch scheduled for release next week.

> [PRIORITY] Thank you for reporting this urgent issue. I've
> escalated the crash to our team for immediate investigation, and
> you can expect a response within four hours during business hours.
> In the meantime, please try restarting the app or updating it from
> the app store, as these steps sometimes resolve crashes.

No passage in the knowledge base mentions a patch next week. The
model invented the patch. The program also escalated nothing, because
it only writes text.

We read the six replies by hand. The four replies at reward 1.0 all
promise an action that the program does not take. The draft
instructions already forbid this: "do not promise anything the
context does not support." The model broke this instruction with and
without a hint. The two `BestOfN` replies to the typo email in §4
make such promises too. The loop then kept the replies that broke
it.

The cause is the reward. A promise of action passes the hard rule and
the `actionable` check. Neither check tests whether the promise is
true.

```text
reply : "... correcting this typo in the upcoming patch scheduled for release
         next week."
  hard rule     the reply has no "support@"      pass
  grounded      the reply names the typo         pass
  actionable    the reply states a next step     pass
  true?         no check asks                    -
reward 1.0
```

The first hint in §5 even suggests a promise: confirm
"investigation steps being initiated."

Session 6 said that an optimizer finds whatever the metric rewards. A
retry loop does the same, one email at a time. Before these replies
go to customers, the reward needs a third check: does every promise
in the reply come from a passage? That check is action item 2.

### Limit 3: an attempt that raises an error counts as an attempt

`limits.py` uses a small test module that fails on purpose and then
answers a sum. The output shows `BestOfN`. `Refine` does the same.

```
--- an attempt that raises ---
BestOfN: Attempt 1 failed with rollout id 0: the model did not answer
1 failure(s) in 3 attempts -> answer: 2 + 2 equals 4.
BestOfN: Attempt 1 failed with rollout id 0: the model did not answer
BestOfN: Attempt 2 failed with rollout id 1: the model did not answer
BestOfN: Attempt 3 failed with rollout id 2: the model did not answer
3 failure(s) in 3 attempts -> raised TimeoutError: the model did not answer
```

Both loops catch an exception, print it, and go to the next attempt.
The failed attempt uses one of the N attempts, and the caller does
not learn that it happened. A loop does protect a step from a
timeout or an output that does not parse. When every attempt raises,
the loop raises the last exception.

### Limit 4: an optimizer cannot change the hint writer

```
--- the predictor tree of the Refine program ---
['triage', 'extract', 'draft.module.predict']
keys in the saved state : ['triage', 'extract', 'draft.module.predict']
```

The wrapped `draft` predictor is in the list. So you can run an
optimizer on a program that already has a loop.

The hint writer is not in the list. `Refine` creates it inside
`forward` on each call, so it is not an attribute of any module. This
is Session 8's rule again: an optimizer changes only predictors that
are attributes of a module. No optimizer can change the
hint writer's instructions, and `save()` stores none of its hints.

## 8. Choosing: Before You Ship, or On Every Call

### The same hint, three times

In §5 the hint writer wrote the same hint for three emails: the
customer is already writing to support. Put a hint that every email
needs in the prompt, where the first draft can use it.

`code/fix-in-prompt.py` adds the hint to the draft signature's
instructions by hand:

```
--- the edited instructions ---
Write a short, courteous reply to a support email. Use the context
passages for any policy, timeline, or known issue you mention, and do not
promise anything the context does not support.
ADDED: The customer is already writing to support. Do not ask them to email support.
       Say what support will do next.
```

Here is the first draft for the charged-twice email, before and after
the edit. Neither program has a loop:

```text
before : "... Please email support@planbook.example with the charge dates, amounts,
          and the last four digits of your card ..."            reward 0.0

after  : "... We will review your account for the duplicate charge and respond
          within two business days to resolve this issue."      reward 1.0
```

The passages did not change. The instructions now tell the model how
to use them.

The script runs the edited program with one draft, and with `Refine`
and larger-model hints, three times each:

```
--- test split, three runs ---
program                     run 1  run 2  run 3   mean  redirects  attempts  hints
one draft, edited prompt     87.5   75.0   66.7   76.4       0/18       1.0    0.0
Refine, edited prompt        87.5   87.5   95.8   90.3       0/18       2.2    1.2
```

Compare with §6:

| Program | Mean score | Redirects | Attempts per email | Hints per email |
|---|---|---|---|---|
| one draft | 73.6 | 8/18 | 1.0 | 0.0 |
| one draft, edited prompt | 76.4 | **0/18** | 1.0 | 0.0 |
| Refine, larger-model hints | 88.9 | 0/18 | 2.5 | 1.5 |
| Refine, edited prompt | 90.3 | 0/18 | 2.2 | 1.2 |

Three short sentences in the instructions removed every redirect, for
no extra calls. The score rose little, because no metric check looks
for redirects. The hints showed which instruction was missing. We added it to the
prompt.

The loop still helps the edited program. Its score rises by about 14
points, and it needs fewer attempts and fewer hints than before. The
edited prompt leaves the loop fewer faults to correct.

One caution: our added sentence "Say what support will do next" also
invites a promise. Action item 2 applies to the edited prompt too.

### Where each fix belongs

| The fault | Example today | Fix | When the fix runs |
|---|---|---|---|
| the same on every input | "email support" | edit the instructions or the knowledge base by hand, or run an optimizer | once, before you ship |
| different on each input, and a new sample often fixes it | a vague first reply to the typo email | `BestOfN` | on every call |
| different on each input, and the model needs to be told | attempt 2 for the charged-twice email, which needed more detail | `Refine` with larger-model hints | on every call |
| no reply can pass | the praise email | hold the reply for a person, and fix the reward | on every call |

Use the loops to find faults as well as to fix them. Read the hints.
When the same hint appears on many inputs, move it into the prompt
and run the loop again.

## 9. Best Practices

### Do

- **Put a free check first.** A line of Python that finds a fault
  costs nothing and never changes its answer. Return before you call
  a judge.
- **Let a hard rule override the other checks.** If a reply must
  never do something, return 0.0 when it does. Do not let it share
  points with the judged checks.
- **Write the reward function for a reader.** `Refine` puts its
  source code in a prompt. Name the checks and say what each one
  means.
- **Give the hint writer the larger model.** Pin the program's
  predictors to the task model with `set_lm`, and select the larger
  model with `dspy.context`. The task model wrote "N/A" for 11 of 12 hints on its own
  replies.
- **Wrap one step, not the whole program.** We wrapped `draft`. The
  labels from `triage` stayed as they were, and each retry ran one
  predictor again, not all of them.
- **Measure more than one run.** One draft moved 12.5 points
  between runs with no change.
- **Check the returned reply yourself.** The loops return a reply at
  any reward. Decide what happens below the threshold.

### Don't

- **Don't use a metric as a reward function.** A metric needs a gold
  example. A reward function runs where none exists.
- **Don't use `BestOfN` against a fault that the prompt causes.**
  Nine attempts on three emails gave nine redirects. Fix the prompt.
- **Don't treat a reward of 1.0 as proof that the reply is correct.** It means only that the
  reply passed your checks. Four such replies promised work that the
  program does not do.
- **Don't score the loop with only the judge that drives it.** A
  second judge agreed on the order of the programs. Check that yours
  does.
- **Don't leave a repeated hint in the loop.** If every input needs
  it, it is an instruction. Put it in the signature.
- **Don't ignore the cost for each call.** A reward function runs on every call.
  Ours added about five calls to the larger model for each email.

## 10. Session Wrap & Next Steps

### Key Takeaways

1. **Retry loops replaced assertions.** `dspy.Assert` and
   `dspy.Suggest` are gone. `BestOfN` and `Refine` wrap a step, score
   each attempt, and try again. A reward function has no gold label,
   so only checks that need none can be rewards.
2. **A new sample can fix a fault from chance. A hint can fix a fault
   from the prompt.**
   Over three runs, `BestOfN` sent 7 redirects in 18 replies.
   `Refine` with larger-model hints sent none.
3. **The hint writer is a hidden predictor.** It uses the configured
   model, it reads your reward function's source code, and no
   optimizer can change it. Its hints are not saved.
4. **A loop keeps the reply with the highest reward.** Our reward checked for a next
   step and no redirect. Four replies met both checks with promises
   that the program does not keep. The
   reward needs a check for truth before these replies go out.
5. **Move a repeated hint into the prompt.** Three short sentences in the instructions removed every redirect
   with one draft. The loop then had less to do.

### Preview of Session 10: Trackers

Today's scripts used workarounds to record what the program did. To
print a hint, we searched DSPy's
global call history for the hint writer's instructions. To count attempts, we wrapped the reward function in a recorder. To find the
cost, we added up one field of every history entry.

Session 10 covers the tools made for this: `dspy.inspect_history`,
usage tracking, callbacks, and tracing with MLflow, so that a running
program reports what it called, what each call cost, and where a
reply came from. Streaming, postponed from Session 8, is there too.

### Action Items

1. Run `code/best-of-n.py` and `code/refine.py`. Open
   `state/refine-attempts.json` and read every attempt for the CSV
   email. Say why no attempt reached 1.0.
2. Add a third judged check to `reply_reward`: does every promise in
   the reply come from a passage? The passages are in
   `args["context"]`. Run `code/compare.py` again and count the
   replies that promise work.
3. Change `reply_reward` so that the praise email can pass. One way:
   skip the `actionable` check when `args["category"]` is
   `feedback`. Report how many replies `send_or_hold` now holds.
4. Run `code/refine.py` with `threshold=0.5`. Report the attempts per email, the hints per email, and the score. Decide which threshold
   you would ship.
5. Wrap the whole `GroundedInbox` in `Refine`, not only the `draft`
   step. The hint writer now reads your `forward` and writes one hint
   for each of the three predictors. Read the hints, and report
   whether any label changed between attempts.
