# MedFlow — a runnable baseline for automated agentic workflow search on clinical calculation

**CSE 598 Agentic AI — capstone proposal baseline (individual).**

The long-term project asks whether an **AFlow-style automated workflow search**
can be applied to **medical calculation**, where a wrong number is a dosing or
risk-stratification error rather than a wrong trivia answer. This repository is
the *baseline* for that project: three hand-written workflows over
[MedCalc-Bench](https://github.com/ncbi-nlp/MedCalc-Bench), a scorer that
matches the benchmark's own convention, and an exact-replay cache so anyone can
reproduce every number below **with no API key and no GPU**.

---

## TL;DR — reproduce the headline result in about 30 seconds

```bash
git clone <this-repo> && cd medflow
python3 run_baseline.py --n 50 --workflow cot --mode offline
```

No install, no key, no network (the dataset ships in `data/`). Expected output
ends with:

```
accuracy            : 56.0%  (28/50)
parse failure rate  : 0.0%
by category         : date=100%(n=3)  diagnosis=33%(n=3)  dosage=0%(n=2)  lab test=65%(n=17)  physical=82%(n=11)  risk=27%(n=11)  severity=33%(n=3)
by output type      : date=100%(n=1)  decimal=72%(n=29)  integer=22%(n=18)  weeks_days=100%(n=2)
llm calls           : 0 live, 50 from cache
```

`--mode offline` replays `runs/cache.jsonl`, the recorded responses from the
live run described below. It is byte-for-byte the same evaluation the live run
performed — only the network call is skipped. The CLI defaults (`--model
qwen3-30b-a3b-instruct-2507 --temperature 0 --max-tokens 1500`) are exactly what
the cache was recorded with, so no extra flags are needed; if you change one,
the run stops with a cache-miss message naming what the cache does cover.

---

## 1. The task

**MedCalc-Bench** (Khandekar et al., NeurIPS 2024 D&B) pairs a real,
de-identified patient note with a clinical-calculator question. The model must
read the needed values out of free text, pick the right formula or rule, and
compute.

* 1,100 test cases, 55 calculators, 7 categories (lab test, risk, physical,
  severity, diagnosis, date, dosage).
* Answers are **verifiable**: decimals are scored against a ±5 % band shipped
  with the dataset, integers exactly, dates on `mm/dd/yyyy`, and gestational age
  as a `(weeks, days)` pair.
* It is genuinely unsolved — the original paper reports GPT-4 at **≈50 %**.

That combination (cheap, deterministic, executable scoring + real headroom +
tool use that actually matters) is what makes it a workable search signal for
the next phase.

**Input** → one patient note + one question.
**Output** → a single value on a final `ANSWER:` line.
**Success** → the value falls inside the benchmark's accepted range.

## 2. What the baseline does

Three workflows, built from a small operator vocabulary so the next phase can
recombine them automatically:

| operator | what it does |
|---|---|
| `op_answer_only` | one call, value only, no reasoning |
| `op_cot` | one call: state the formula, list the values read from the note, convert units, compute |
| `op_extract` | one call that only pulls the needed clinical values out of the note |
| `op_program` | one call that emits Python, executed in a restricted sandbox |

| workflow | composition |
|---|---|
| `direct` | `op_answer_only` |
| **`cot`** | `op_cot` — **this is the baseline** |
| `pot` | `op_extract` → `op_program` → execute, falling back to `op_cot` if the code fails |

Every workflow has the same signature, `run(client, row) -> {"answer_text", "trace"}`,
and is scored by the same evaluator. That is the seam the automated search
plugs into: a searcher emits a new `run` body over the same operators and the
existing harness scores it unchanged.

## 3. Measured results

Model `qwen3-30b-a3b-instruct-2507` on ASU's OpenAI-compatible gateway,
temperature 0, 50 cases stratified to match the full test set's category mix
(`--n 50 --seed 0`).

```
workflow       acc  parse-fail   calls    tokens    llm_s
----------------------------------------------------------
direct      48.0%        0.0%      50     55668    193.8
cot         56.0%        0.0%      50     66150    286.6
pot         58.0%        0.0%     106    114575    377.2
```

`llm_s` is server-side latency summed over calls, taken from the recording, so
it is the same number live or replayed and does not depend on `--workers`. Wall
clock for the live 50-case `cot` run at `--workers 8` was 44 s.

Per category, **no single workflow wins everywhere**:

```
category       n   direct      cot      pot       best
-------------------------------------------------------
date           3     67%    100%    100%        cot
diagnosis      3     33%     33%     33%     direct
dosage         2     50%      0%     50%     direct
lab test      17     59%     65%     65%        cot
physical      11     64%     82%     73%        cot
risk          11     27%     27%     45%        pot
severity       3      0%     33%      0%        cot

best single workflow          : 58.0%
best workflow per category    : 62.0%   (+4.0% vs best single)
any-workflow-correct (oracle) : 74.0%   (+16.0% vs best single)
```

Reproduce this table with:

```bash
python3 scripts/compare.py
```

The gap between 58 % and 62–74 % is the headroom the capstone's automated
search is aiming at, and it is measured rather than assumed. The other clear
signal is the split by answer type: **decimal 72 % vs integer 22 %**. The
integer cases are the rule-based severity and risk indices (Caprini, GCS,
CHA₂DS₂-VASc, SOFA), where the model must apply a checklist of criteria rather
than evaluate one formula — a different failure mode that plausibly wants a
different workflow.

## 4. Concrete test case

`examples/test1.json` is case `row1`: Creatinine Clearance (Cockcroft-Gault),
where the question additionally requires deciding between actual and adjusted
body weight based on BMI. Ground truth `25.2381`, accepted band
`[23.97619, 26.50001]`.

```bash
python3 run_baseline.py --input examples/test1.json --mode offline
```

```
MedFlow baseline | workflow=cot model=qwen3-30b-a3b-instruct-2507 mode=offline cases=1
------------------------------------------------------------------------------
[  1/1] PASS  row1     pred=25.0         gt=25.2381      Creatinine Clearance (Cockcroft-Gault
------------------------------------------------------------------------------
accuracy            : 100.0%  (1/1)
parse failure rate  : 0.0%
llm calls           : 0 live, 1 from cache
```

The full model reasoning is in
`runs/cot_qwen3-30b-a3b-instruct-2507_n1.traces.json`. It is worth reading,
because the case **passes for the wrong reason**: the model derives 25.238
correctly, then adds a "Step 5: Round to the nearest whole number (standard
practice for CrCl)" and answers `25`. It lands inside the tolerance band only
because the band is ±5 %. On a tighter-tolerance calculator the same premature
rounding is a failure — which is exactly the kind of defect a `verify` or
`recompute` operator in the searched workflow should catch.

`examples/test3.json` covers one case per answer format (decimal, integer,
gestational-age tuple).

## 4b. A note on determinism, and why the cache exists

**Temperature 0 does not make this endpoint bit-exact.** Re-running the same
50-case `cot` command live a second time scored **54.0 %**, not 56.0 % — three
cases flipped. The transcript of that second run is kept in
`docs/screenshot_live_rerun.txt` rather than hidden, because it is the whole
argument for shipping a cache: a workflow-search project that compares
candidate workflows to each other cannot tell a 2-point search improvement from
a 2-point sampling wobble unless the comparison is pinned.

Two properties keep replay stable:

* **First write wins.** A live re-run appends a fresh response for an
  already-recorded request; `--mode offline` keeps replaying the *first*
  recording, so the reported numbers do not drift as the cache grows.
  `python3 scripts/dedupe_cache.py` prunes the superseded lines.
* **The pipeline itself is deterministic.** `scripts/selftest.py` replays the
  `pot` workflow at `--workers 1`, `4` and `8` and requires all three to score
  identically. (An early version failed this: the sandbox read its result queue
  with `join()`-then-`empty()`, so a child that exited before its write was
  drained was misread as a crash and spuriously fell back to CoT.)

For the project itself the implication is that a searched workflow must be
scored on the same pinned cases, and any claimed gain smaller than the sampling
wobble needs repeated sampling before it counts.

## 5. Running it live

Any OpenAI-compatible endpoint works — ASU's gateway, OpenAI, or a local vLLM
server.

```bash
pip install -r requirements.txt        # only `openai`, and only needed for live calls

export OPENAI_BASE_URL=https://openai.rc.asu.edu/v1     # or https://api.openai.com/v1
export OPENAI_API_KEY=...                               # ASU: $(cat ~/.asu_llm_key)

python3 run_baseline.py \
    --n 50 --workflow cot --mode auto \
    --model qwen3-30b-a3b-instruct-2507 \
    --max-tokens 1500 --workers 8
```

`--mode auto` serves anything already cached and calls the API only for misses,
so re-running costs nothing. New responses are appended to `runs/cache.jsonl`.

### Modes

| `--mode` | behaviour | needs a key? |
|---|---|---|
| `offline` *(default)* | cache only; a miss is a hard error with a clear message | no |
| `auto` | cache first, API on miss, append to cache | yes |
| `live` | always call the API | yes |
| `mock` | deterministic fake LLM fed the ground truth; plumbing check only | no |

### Useful flags

| flag | meaning |
|---|---|
| `--workflow {direct,cot,pot}` | which workflow (default `cot`) |
| `--n N --seed S` | stratified subset, deterministic for a given `(N, S)` |
| `--input FILE` | run explicit case ids instead (see `examples/`) |
| `--workers K` | parallel requests, default 4; results stay in case order |
| `--disable-thinking` | sends vLLM's `chat_template_kwargs.enable_thinking=false` |
| `--max-tokens N` | completion budget, default 1024 |

### Choosing a model

Use an **instruct** model, not a thinking one. A hybrid reasoning model such as
`qwen35-27b` spends the whole completion budget in `reasoning_content` and
returns `content: None` with `finish_reason="length"` — which reads as a broken
endpoint. The client detects that case and says so explicitly rather than
scoring an empty string:

```
[  1/1] FAIL  row1     pred=None    gt=25.2381    Creatinine Clearance (Cockcroft-Gault
```
```
[NO CONTENT: finish_reason=length; the model returned only reasoning_content
 (4795 chars). Raise --max-tokens or pass --disable-thinking.]
```

Either raise `--max-tokens` or pass `--disable-thinking`.

## 6. Verifying the install

```bash
python3 scripts/selftest.py
```

Checks the scorer on all four answer formats, the sandbox (arithmetic, blocked
`import`, killed infinite loop), the dataset checksum and subset determinism,
and then runs all three workflows end-to-end against a mock LLM that is fed the
ground truth — every case must come back correct. Finally it replays the `pot`
workflow at three different `--workers` settings and requires an identical
score. Takes about a minute and needs no key. Ends with `SELFTEST PASSED`.

## 7. Layout

```
run_baseline.py             CLI entry point
medflow/
  data.py                   download + checksum + deterministic stratified subset
  llm.py                    OpenAI-compatible client, cache/replay, reasoning-model quirks
  workflows.py              operators and the three workflows
  evaluate.py               answer parsing and MedCalc-Bench scoring
  sandbox.py                restricted exec for the `pot` workflow
  mock.py                   fake LLM for the self-test
scripts/
  selftest.py               offline correctness checks + replay determinism
  compare.py                cross-workflow table + headroom ceilings
  dedupe_cache.py           prune superseded cache recordings
examples/
  test1.json                single concrete test case (row1)
  test3.json                one case per answer format
data/test_data.csv          MedCalc-Bench test split, sha256-pinned
runs/
  cache.jsonl               recorded LLM responses -> exact offline replay
  *_n50.json                per-case results + summary for each workflow
  *_n50.traces.json         full prompts, replies and executed code
docs/
  screenshot_test1.txt      the single concrete test case
  screenshot_baseline50.txt the 50-case baseline
  screenshot_selftest.txt   the self-test
  screenshot_compare.txt    the cross-workflow table
  screenshot_live_rerun.txt a second LIVE run, scoring 54.0% (see section 4b)
```

## 8. Requirements and limitations

* **Python 3.9+**, Linux or macOS. Offline and mock modes use only the standard
  library; `openai>=1.40` is needed only for live calls.
* `data/test_data.csv` is committed (5.1 MB) and sha256-pinned. If absent it is
  downloaded from the NCBI GitHub mirror. The HuggingFace copy of MedCalc-Bench
  is gated and is deliberately **not** used, so nothing here needs an approval
  step.
* `runs/cache.jsonl` covers exactly the configurations in this README, which
  are also the CLI defaults: `qwen3-30b-a3b-instruct-2507`, temperature 0,
  `--max-tokens 1500`, the three workflows at `--n 50 --seed 0`, plus
  `examples/test1.json` and `examples/test3.json`. Changing the model,
  temperature, token budget or any prompt changes the cache key, so
  `--mode offline` will report a clear cache miss rather than silently returning
  a stale answer. That is intentional.
* The `pot` sandbox blocks `import`/`exec`/`open` and kills the process after
  5 s. It is a guard against a confused model, **not** a security boundary
  against an adversary; the only code it ever runs is what our own prompt
  produced against a public benchmark.
* 50 cases is a small sample — 56 % ± roughly 7 pp at one standard error, and a
  live re-run moved 2 points on its own (section 4b). The
  per-category cells (n = 2–17) are directional, not conclusive. Scaling to the
  full 1,100 is one command (`--n 1100`); it is left out of the baseline to keep
  the reproduction fast.
* Results are from one open-weight 30B model on one gateway. Absolute numbers
  will move with the model; the per-category *disagreement* between workflows is
  the part the project depends on.

## 9. Next phase

Replace the three hand-written workflows with an AFlow-style search that
composes the operator vocabulary automatically, scored by this same evaluator
with cost and abstention folded into the objective alongside accuracy. The
harness, the scorer, the stratified split and the cost accounting are already in
place; only the searcher is missing.

## Citation

MedCalc-Bench: Khandekar, N. et al. *MedCalc-Bench: Evaluating Large Language
Models for Medical Calculations.* NeurIPS 2024 Datasets & Benchmarks.
<https://github.com/ncbi-nlp/MedCalc-Bench>

AFlow: Zhang, J. et al. *AFlow: Automating Agentic Workflow Generation.*
ICLR 2025 (Oral). <https://arxiv.org/abs/2410.10762>
