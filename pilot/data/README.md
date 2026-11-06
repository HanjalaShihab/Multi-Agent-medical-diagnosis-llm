# Data notes

All loaders turn a public benchmark into one JSONL file of `Case` objects. The
pilot uses **only human-written benchmarks**: MedQA, PubMedQA (`pqa_labeled`),
MMLU-Med, MedMCQA (validation), and MedXpertQA (text). No synthetic,
constructed, or LLM-restructured dataset is used.

## Case schema

A case has:

```json
{
  "case_id": "medqa-00000",
  "question": "Which single option is the most likely answer?",
  "options": ["Methadone", "Naloxone", "Loperamide", "Alvimopan"],
  "answer": 0,
  "evidence_slices": {
    "part_1": "A 34-year-old man ...",
    "part_2": "He has been discontinuing heroin ...",
    "part_3": "He is HIV positive ...",
    "part_4": "He reports pain ...",
    "part_5": "To comprehensively treat these symptoms, ..."
  },
  "rarity_stratum": null,
  "prevalence": null,
  "metadata": {
    "source": "medqa",
    "slice_kind": "text_chunks",
    "gold_slice": "part_1"
  }
}
```

`question` is the instruction. The evidence lives in `evidence_slices` so that
H4 can give each agent a disjoint part of the record. `rarity_stratum` and
`prevalence` are present for schema compatibility but are **`None` for every
real benchmark in this suite**.

PubMedQA uses real abstract sections instead of text chunks:

```json
{
  "case_id": "pubmedqa-17342562",
  "question": "The clinical significance of bile duct sludge: is it different from bile duct stones? Answer yes if the evidence supports the claim, no if it refutes it, and maybe if it is inconclusive.",
  "options": ["no", "yes", "maybe"],
  "answer": 0,
  "evidence_slices": {
    "background": "Some patients with suspected common bile duct stones ...",
    "methods": "We reviewed ...",
    "results": "Sludge was found in ...",
    "conclusions": "Bile duct sludge ..."
  },
  "rarity_stratum": null,
  "prevalence": null,
  "metadata": {
    "source": "pubmedqa",
    "slice_kind": "abstract_sections",
    "section_labels": ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"],
    "gold_slice": "results"
  }
}
```

## Quick download

```bash
cd pilot
python scripts/download_data.py all --out data/raw
```

The downloader uses public Hugging Face `resolve` URLs and the Python standard
library, so `huggingface_hub` is not required. To fetch one benchmark, pass its
name (`medqa`, `pubmedqa`, `mmlu`, `medmcqa`, `medxpertqa`).

## Datasets

| Dataset | Prepare command | Use |
|---|---|---|
| **MedQA-USMLE** | `prepare --dataset medqa --raw data/raw/medqa` | 4-option USMLE-style QA; H1–H3, SYC. |
| **PubMedQA** | `prepare --dataset pubmedqa --raw data/raw/pubmedqa` | 3-way yes/no/maybe; the only genuine H4 evidence partition. |
| **MMLU-Med** | `prepare --dataset mmlu_med --raw data/raw/mmlu` | Six medical subjects; H1–H3, SYC. |
| **MedMCQA** | `prepare --dataset medmcqa --raw data/raw/medmcqa` | Validation split; broad exam QA. |
| **MedXpertQA** | `prepare --dataset medxpertqa --raw data/raw/medxpertqa` | Expert-level QA with up to 10 options. |

`generic` is an optional escape hatch for a user-supplied file. It is not part
of the pilot suite.

### MedQA-USMLE

- **Source:** `GBaker/MedQA-USMLE-4-options`, file
  `phrases_no_exclude_test.jsonl`.
- **License:** CC-BY-4.0 (dataset card).
- **Rows:** 1,273 in the test split.
- **Citation:** Jin et al., 2021, *What Disease does this Patient Have? A
  Large-scale Open Domain Question Answering Dataset from Medical Exams*.
- **Raw layout:**

  ```
  data/raw/medqa/
    test.jsonl
  ```

The loader reads `question`, the `options` dict (`A`–`D`), and `answer_idx`. It
replaces the question with a fixed instruction and splits the original stem into
five contiguous text chunks (`part_1`..`part_5`). Slices are not clinical
sections; H4 here is a text-chunk ablation.

### PubMedQA (human-annotated only)

- **Source:** `qiaojin/PubMedQA`, `pqa_labeled/train-00000-of-00001.parquet`.
- **License:** MIT.
- **Rows:** 1,000 human-annotated questions.
- **Citation:** Jin et al., 2019, *PubMedQA: A Dataset for Biomedical Research
  Question Answering*.
- **Raw layout:**

  ```
  data/raw/pubmedqa/
    pqa_labeled.parquet
  ```

`pqa_artificial` is deliberately rejected: those 211k rows are generated from
unlabeled abstracts and are not human-written benchmark items. The loader
accepts only `pqa_labeled` and raises an error if the filename contains
`artificial`.

The three options are `yes`, `no`, and `maybe`, taken from `final_decision`. The
context labels are free text and highly variable. The loader buckets them into
six disjoint sections:

| Bucket | Typical source labels |
|---|---|
| `background` | BACKGROUND, OBJECTIVE, PURPOSE, AIM, CONTEXT, INTRODUCTION |
| `case` | CASE REPORT, CASE PRESENTATION, CASE DESCRIPTION |
| `methods` | METHODS, MATERIALS, DESIGN, SETTING, PARTICIPANTS, INTERVENTION, MEASURE |
| `results` | RESULTS, FINDINGS, OBSERVATIONS |
| `conclusions` | CONCLUSIONS, DISCUSSION, LIMITATIONS, INTERPRETATION |
| `other` | anything unmatched |

The original labels are kept in `metadata.section_labels`. This is the **only**
benchmark in the suite with genuine, human-written evidence sections, so it is
the only one where H4/H5 test actual evidence partitioning. Many abstracts have
only 2–4 sections after bucketing; when there are fewer sections than agents, the
runner reuses sections and prints a warning. Report that reuse rate.

### MMLU-Med

- **Source:** `cais/mmlu`, test split for six medical subjects.
- **License:** MIT.
- **Rows:** anatomy 135, clinical knowledge 265, college medicine 173, medical
  genetics 100, professional medicine 272, virology 166 (1,111 total).
- **Citation:** Hendrycks et al., 2021, *Measuring Massive Multitask Language
  Understanding*.
- **Raw layout:**

  ```
  data/raw/mmlu/
    anatomy_test.parquet
    clinical_knowledge_test.parquet
    college_medicine_test.parquet
    medical_genetics_test.parquet
    professional_medicine_test.parquet
    virology_test.parquet
  ```

The loader reads `question`, the `choices` array, and the integer `answer`. It
samples across all six subjects by default; pass `--subjects` to select a
subset. The stem is split into five text chunks, so H4 is a text-chunk ablation.

### MedMCQA

- **Source:** `openlifescienceai/medmcqa`, `validation.parquet`.
- **License:** Apache-2.0.
- **Rows:** 4,183 labelled validation rows.
- **Citation:** Pal et al., 2022, *MedMCQA: A Large-scale Multi-Subject
  Multi-Choice Dataset for Medical domain Question Answering*.
- **Raw layout:**

  ```
  data/raw/medmcqa/
    validation.parquet
  ```

The official **test** split ships `cop == -1` for every row because the labels
are withheld, so it cannot be scored locally. The loader uses the validation
split, the public labelled split, and refuses a file whose labels are all
missing. This deviation from the paper's test-set protocol must be stated in any
write-up.

Options come from `opa`..`opd`; `cop` is the 0–3 answer index. The subject and
topic are stored in metadata. The stem is split into five text chunks.

### MedXpertQA

- **Source:** `TsinghuaC3I/MedXpertQA`, `Text/test.jsonl`.
- **License:** MIT.
- **Rows:** 2,450 text items.
- **Citation:** Zuo et al., 2025, *MedXpertQA: Benchmarking Expert-Level Medical
  Reasoning and Understanding*.
- **Raw layout:**

  ```
  data/raw/medxpertqa/
    test.jsonl
  ```

Each row has `id`, `question`, `options` (`A`–`J`), `label` (`A`–`J`), and
metadata (`medical_task`, `body_system`, `question_type`). The question text
repeats all options inline after "Answer Choices:". The loader strips that block
so the options are not duplicated in the prompt. It keeps up to ten options by
default. The stem is split into five text chunks.

The dataset also ships a multimodal split; this pilot uses the text-only file.

## Answer handling

- `_options_from_row` accepts a dict, a list, a NumPy/pandas array, or MedMCQA's
  `opa`–`opd`.
- `_answer_index` accepts an integer index, a numeric string, a single letter, or
  the exact option text.
- `_order_options` keeps the gold option, caps the list at `max_options`
  (default 10), and shuffles deterministically from the case id and seed.
- Cases with fewer than two options are skipped.

## Limitations that belong in the paper

1. **No disease prevalence.** Real QA benchmarks do not encode a
   patient-population prevalence prior. `rarity_stratum` and `prevalence` are
   `None`; they are **never** derived from answer-option frequency. Answer
   frequency is a property of the benchmark answer key, not of disease
   prevalence. P6a/P6b are mechanical properties of the BTS score; claims about
   disease rarity are not testable on this suite.
2. **Only PubMedQA has real evidence sections.** MedQA, MMLU-Med, MedMCQA, and
   MedXpertQA are text-only; their five slices are contiguous chunks of the
   question stem. H4/H5 on those datasets are a text-chunk ablation, not
   evidence partitioning.
3. **Section reuse in PubMedQA.** After bucketing, many abstracts have fewer
   than five sections. With five agents, sections repeat. Report the reuse rate.
4. **MedMCQA uses validation, not test.** The public test split withholds labels.
5. **Benchmark accuracy is not clinical validity.** These are exam-style items,
   not patient records. The pilot is a measurement study, not a diagnostic
   device.

## Licenses and provenance

Record the download date and the analysis code version in your paper appendix.

| Dataset | License | Citation |
|---|---|---|
| MedQA-USMLE | CC-BY-4.0 | Jin et al., 2021 |
| PubMedQA (`pqa_labeled`) | MIT | Jin et al., 2019 |
| MMLU | MIT | Hendrycks et al., 2021 |
| MedMCQA | Apache-2.0 | Pal et al., 2022 |
| MedXpertQA | MIT | Zuo et al., 2025 |

BioASQ is not included. The public mirrors are gated or do not ship the benchmark
file; download it manually from <https://bioasq.org/> only if you have access.

## Optional generic loader

`--dataset generic` reads a user-supplied JSON/JSONL/CSV file with a question, an
option list, and an answer. Use a field map when the column names differ:

```bash
python -m bts_audit.cli prepare \
  --dataset generic \
  --raw data/raw/my_cases.json \
  --out data/cases/my_cases.jsonl \
  --field-map '{"question": ["case_presentation", "question"], "options": ["differential"], "answer": ["final_diagnosis"]}'
```

If the file has no option list, the loader builds one from the gold answer plus
other answers in the file. That is a fallback, not a real differential. The
generic loader is **not** part of the pilot suite.

## Synthetic and constructed data are not evidence

The pilot does not ship or use synthetic, constructed, or LLM-restructured
datasets. Do not report mock-backend numbers as a scientific result;
`report.md` marks mock runs as `SMOKE ONLY`.
