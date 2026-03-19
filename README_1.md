# BioClinicalModernBertChecker

A context-sensitive neural spell checker for clinical free text, built on
[`thomas-sounack/BioClinical-ModernBERT-base`](https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-base)
and modelled on the architecture of the `BertChecker` from the
[NeuSpell](https://github.com/neuspell/neuspell) toolkit.

---

## Motivation

Spelling errors in clinical documentation are common, consequential, and poorly
served by general-purpose spell checkers. Clinicians write under time pressure,
using specialised terminology, Latin abbreviations, drug names, and domain
shorthand that standard English dictionaries do not recognise. Over-correction
of legitimate clinical abbreviations (`nad`, `hpi`, `pcr`) is as harmful as
failing to correct genuine misspellings, because it introduces new errors into
the medical record.

NeuSpell's `BertChecker` (Jayanthi et al., EMNLP 2020) established a strong
baseline for context-sensitive neural spelling correction, but it was designed
for general English text and trained on the BEA-60K dataset. Its implementation
also relies on several dependencies that have since been deprecated — notably
`pytorch_pretrained_bert.BertAdam`, a hardcoded WordPiece tokeniser, and a
global mutable tokeniser state that makes multi-instance use unsafe.

This project adapts and modernises the `BertChecker` architecture specifically
for clinical text, replacing the general-domain BERT backbone with a
domain-trained clinical encoder and updating the implementation throughout for
compatibility with current `transformers` and `torch` APIs.

---

## Why BioClinical-ModernBERT?

[`BioClinical-ModernBERT-base`](https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-base)
(Sounack et al., 2025) is a masked language model fine-tuned on biomedical and
clinical corpora from the base
[ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) architecture.
It was chosen over alternatives for three reasons:

**Domain fit.** The model was trained on clinical and biomedical text, meaning
its token representations encode the vocabulary and co-occurrence statistics of
the domain where the spell checker will be deployed. A general BERT model
trained on Wikipedia and BookCorpus does not know that `tachypnea` is more
likely than `tachypoea` in context, or that `metoprolol` and `carvedilol`
occupy similar distributional positions.

**Long-context support.** ModernBERT supports a context window of up to 8,192
tokens, compared to 512 for standard BERT. Clinical notes — particularly
discharge summaries — regularly exceed 512 tokens. Previous BERT-based spell
checkers silently truncate long notes, discarding the contextual signal that
makes neural correction superior to dictionary lookup.

**BPE tokenisation.** ModernBERT uses Byte-Pair Encoding rather than WordPiece.
The original NeuSpell subword merging logic counted `##`-prefixed tokens to
reconstruct word-level predictions, which is hardcoded to WordPiece and breaks
silently with any other tokeniser. This implementation uses the
`word_ids()` API, which is tokeniser-agnostic and correctly handles BPE, 
WordPiece, and Unigram tokenisers alike.

**Task alignment.** The `fill-mask` (`AutoModelForMaskedLM`) task head is
preserved in `BioClinical-ModernBERT-base`, making it directly usable for
token-level sequence labelling. 

---

## Differences from the original NeuSpell BertChecker

| Aspect | NeuSpell BertChecker (2020) | This implementation |
|---|---|---|
| Backbone | `bert-base-cased` | `BioClinical-ModernBERT-base` |
| Context window | 512 tokens | 8,192 tokens |
| Tokeniser handling | WordPiece `##` counting (hardcoded) | `word_ids()` (tokeniser-agnostic) |
| Tokeniser state | Global mutable (thread-unsafe) | Instance attribute |
| Optimiser | Deprecated `BertAdam` | `torch.optim.AdamW` |
| LR schedule | None | Linear warmup + decay |
| Mixed precision | Not supported | `torch.amp.autocast` + `GradScaler` |
| Gradient clipping | Not supported | `clip_grad_norm_` |
| OOV handling | Raises error | Backoff to original input token |
| `token_type_ids` | Always passed | Conditionally omitted (ModernBERT does not use them) |
| Checkpoint metadata | Not saved | `encoder_name.txt` saved with model |
| Python dependency | Requires `neuspell` | Standalone; `neuspell` optional |

---

## Installation

```bash
pip install torch transformers tqdm
```

No other dependencies are required. If `neuspell` is installed it will be used
for its noising utilities, but all core functionality is self-contained.

---

## Usage

### Quick start

```python
from bioclinical_modernbert_checker import BioClinicalModernBertChecker, build_vocab_from_files

# Build vocabulary from your clinical corpus (clean + synthetically noised)
vocab = build_vocab_from_files(
    clean_file="train_clean.txt",
    corrupt_file="train_noisy.txt",
    data_dir="data/",
    keep_simple=False,   # preserve clinical abbreviations and terminology
    min_freq=2,
    topk=100_000,
)

# Initialise and load pretrained weights
checker = BioClinicalModernBertChecker(device="cuda")
checker.from_huggingface(vocab=vocab)

# Fine-tune on a parallel clinical corpus
checker.finetune(
    clean_file="train_clean.txt",
    corrupt_file="train_noisy.txt",
    data_dir="data/",
    n_epochs=3,
    learning_rate=5e-5,
)

# Correct a single string
checker.correct("Pt c/o worsning dyspenea and fiver")
# → "Pt c/o worsening dyspnea and fever"

# Correct a batch
checker.correct_strings([
    "Patient denies nausea, vomting, or diarreah.",
    "Lungs cleer to ausculattion bilaterally.",
])

# Correct from file
checker.correct_from_file(src="noisy_notes.txt", dest="corrected_notes.txt")
```

### Loading a saved model

```python
checker = BioClinicalModernBertChecker(device="cpu")
checker.from_pretrained("path/to/saved/model/")
checker.correct("Afebrile, vitals satble, no acute distrss.")
```

### Evaluation

```python
checker.evaluate(
    clean_file="test_clean.txt",
    corrupt_file="test_noisy.txt",
    data_dir="data/",
)
```

---

## Preparing training data

Clinical notes should be treated as the **clean** side of the training pair.
Synthetic noise is injected at training time using character-level perturbations
(substitution, deletion, insertion, transposition) drawn from a keyboard
adjacency model. Only purely alphabetic tokens of three or more characters are
eligible for corruption; clinical abbreviations, numeric strings, time
expressions, and lab values are protected by a regex guard.

---

## Evaluation

The checker was evaluated on 100 unseen MIMIC-IV-EXT-BHC discharge note segments
using synthetically injected character-level noise (substitution, deletion, insertion,
and transposition) applied to a held-out test set with a random seed distinct from
training. Token-level metrics were computed against gold-standard clean text with
punctuation normalisation to avoid penalising cases where spelling was correctly
restored but trailing punctuation was dropped.

| Metric | Score |
|---|---|
| Word Correction Rate (Recall) | 0.7946 |
| Precision | 0.9018 |
| F1 | 0.8448 |
| False Positive Rate | 0.0085 |

Of 1,582 tokens examined, 1,157 misspellings were correctly restored (TP), 226 were
missed (FN_missed), and 73 were corrected to the wrong target (FN_wrong). The false
positive rate of 0.85% reflects the residual over-correction of short in-vocabulary
clinical abbreviations (`nad`, `hpi`, `cad`, `mri`) that are
orthographically similar to common English words. These are mitigated at inference
time by an explicit clinical abbreviation protection list and a post-inference
restoration guard; the remaining false positives are concentrated in this
abbreviation class rather than in general clinical vocabulary.

> **Note:** These results reflect evaluation on synthetically corrupted text.
> Performance on naturally occurring clinical misspellings may differ; the
> [ClinSpell benchmark](https://github.com/clips/clinspell) (Fivez et al., 2017)
> provides a standard reference for comparison against other clinical spell checkers.


## Citations

**NeuSpell**

```bibtex
@inproceedings{jayanthi-etal-2020-neuspell,
    title     = "{N}eu{S}pell: A Neural Spelling Correction Toolkit",
    author    = "Jayanthi, Sai Muralidhar and Pruthi, Danish and Neubig, Graham",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods
                 in Natural Language Processing: System Demonstrations",
    month     = oct,
    year      = "2020",
    address   = "Online",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2020.emnlp-demos.21",
    doi       = "10.18653/v1/2020.emnlp-demos.21",
    pages     = "158--164",
}
```

**BioClinical-ModernBERT**

```bibtex
@misc{sounack2025bioclinicalmodernbert,
    title         = "{BioClinical-ModernBERT}: A Modern Clinical Encoder",
    author        = "Sounack, Thomas and others",
    year          = "2025",
    eprint        = "2506.10896",
    archivePrefix = "arXiv",
    primaryClass  = "cs.CL",
    url           = "https://arxiv.org/abs/2506.10896",
}
```


## Licence

MIT
