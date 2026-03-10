"""
bioclinical_modernbert_checker.py
==================================
A modernised neural spell-checker that uses BioClinical-ModernBERT-base
(thomas-sounack/BioClinical-ModernBERT-base) as its encoder in place of the
vanilla bert-base-cased used by NeuSpell's BertChecker.

Designed as a drop-in replacement for ``neuspell.BertChecker`` — it exposes
the same public API::

    checker = BioClinicalModernBertChecker()
    checker.from_huggingface(vocab=vocab)          # fresh model
    # or
    checker.from_pretrained(ckpt_path="path/to/ckpt")  # saved checkpoint

    checker.correct("Pt c/o worsning dyspenea and fiver")
    checker.correct_strings([...])
    checker.correct_from_file(src="noisy.txt")
    checker.evaluate(clean_file="clean.txt", corrupt_file="noisy.txt", data_dir=".")
    checker.finetune(clean_file="clean.txt", corrupt_file="noisy.txt", data_dir=".")

Key improvements over NeuSpell's BertChecker (as of the 2021 codebase)
-----------------------------------------------------------------------
1.  **Tokenizer-agnostic subword merging** — uses HuggingFace `word_ids()`
    instead of `##`-prefix counting, so any fast tokenizer works (BPE,
    WordPiece, Unigram …).

2.  **No global mutable tokenizer state** — the tokenizer is an instance
    attribute; multiple checkers can coexist safely.

3.  **Modern optimiser stack** — ``torch.optim.AdamW`` + HuggingFace linear
    warmup/decay scheduler; ``pytorch_pretrained_bert.BertAdam`` is removed.

4.  **Automatic Mixed Precision (AMP)** — ``torch.amp.autocast`` / GradScaler
    on CUDA; graceful fallback on CPU.

5.  **Gradient clipping** — prevents gradient explosions during fine-tuning on
    small clinical datasets.

6.  **Vocabulary-aware OOV back-off** — unknown output tokens are replaced with
    the original input token (pass-through), protecting clinical terminology
    that is absent from the training vocabulary.

7.  **token_type_ids guard** — omits that key if the model config says it is
    unused (ModernBERT does not use it; BERT does).

8.  **Quantisation support** — ``quantize_model()`` inherited from the base.

Dependencies
------------
    pip install torch transformers tqdm

Optional (for full NeuSpell compatibility):
    pip install neuspell
"""

from __future__ import annotations

import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
# Optional NeuSpell integration
# ---------------------------------------------------------------------------
try:
    from neuspell.corrector import Corrector as _NeuSpellBase
    from neuspell.seq_modeling.helpers import (
        batch_accuracy_func,
        batch_iter,
        get_model_nparams,
        get_tokens,
        labelize,
        load_data,
        load_vocab_dict,
        save_vocab_dict,
        train_validation_split,
        untokenize_without_unks,
    )
    from neuspell.seq_modeling.evals import get_metrics
    _NEUSPELL_AVAILABLE = True
except ImportError:
    _NEUSPELL_AVAILABLE = False
    warnings.warn(
        "neuspell not found — NeuSpell helper functions will be inlined. "
        "Install it with: pip install neuspell",
        ImportWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Inline fallbacks (used only when neuspell is not installed)
# ---------------------------------------------------------------------------
if not _NEUSPELL_AVAILABLE:
    import pickle
    import numpy as np
    from torch.nn.utils.rnn import pad_sequence

    def load_vocab_dict(path_: str) -> dict:
        with open(path_, "rb") as fp:
            return pickle.load(fp)

    def save_vocab_dict(path_: str, vocab_: dict) -> None:
        with open(path_, "wb") as fp:
            pickle.dump(vocab_, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(base_path: str, corr_file: str, incorr_file: str) -> List[Tuple[str, str]]:
        def _read(p):
            with open(p, encoding="utf-8") as f:
                return [l.strip() for l in f if l.strip()]
        corr = _read(os.path.join(base_path, corr_file))
        incorr = _read(os.path.join(base_path, incorr_file))
        assert len(corr) == len(incorr), "clean/corrupt file length mismatch"
        return list(zip(corr, incorr))

    def train_validation_split(data, train_ratio, seed):
        np.random.seed(seed)
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        cut = int(np.ceil(train_ratio * len(data)))
        return [data[i] for i in idx[:cut]], [data[i] for i in idx[cut:]]

    def batch_iter(data, batch_size, shuffle):
        indices = list(range(len(data)))
        if shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(data), batch_size):
            batch_idx = indices[i: i + batch_size]
            yield [data[j][0] for j in batch_idx], [data[j][1] for j in batch_idx]

    def labelize(batch_labels, vocab):
        t2i = vocab["token2idx"]
        pad_id = t2i[vocab["pad_token"]]
        unk_id = t2i[vocab["unk_token"]]
        seqs = [[t2i.get(tok, unk_id) for tok in line.split()] for line in batch_labels]
        tensors = [torch.tensor(s) for s in seqs]
        padded = pad_sequence(tensors, batch_first=True, padding_value=pad_id)
        lengths = torch.tensor([len(s) for s in seqs]).long()
        return padded, lengths

    def untokenize_without_unks(batch_preds, batch_lengths, vocab, batch_clean, backoff="pass-through"):
        i2t = vocab["idx2token"]
        unk_id = vocab["token2idx"][vocab["unk_token"]]
        result = []
        for preds, length, clean in zip(batch_preds, batch_lengths, batch_clean):
            clean_words = clean.split()
            tokens = [
                i2t[idx] if idx != unk_id else clean_words[i]
                for i, idx in enumerate(preds[:length])
            ]
            result.append(" ".join(tokens))
        return result

    def batch_accuracy_func(preds, targets, lengths):
        count_, total_ = 0, 0
        for p, t, l in zip(preds, targets, lengths):
            count_ += (p[:l] == t[:l]).sum()
            total_ += l
        return count_, total_

    def get_model_nparams(model):
        return sum(p.numel() for p in model.parameters())

    def get_tokens(data, keep_simple=False, min_max_freq=(1, float("inf")), topk=None):
        from math import log
        freq = {}
        for ex in tqdm(data, desc="building vocab"):
            for tok in ex.split():
                freq[tok] = freq.get(tok, 0) + 1
        if keep_simple:
            freq = {t: f for t, f in freq.items() if t.isascii() and not any(c.isdigit() for c in t)}
        lo, hi = min_max_freq
        freq = {t: f for t, f in freq.items() if lo <= f <= hi}
        if topk:
            freq = dict(sorted(freq.items(), key=lambda x: -x[1])[:topk])
        pad, unk = "<pad>", "<unk>"
        token2idx = {pad: 0, unk: 1}
        for tok in freq:
            if tok not in token2idx:
                token2idx[tok] = len(token2idx)
        idx2token = {v: k for k, v in token2idx.items()}
        return {"token2idx": token2idx, "idx2token": idx2token,
                "token_freq": freq, "pad_token": pad, "unk_token": unk}

    def get_metrics(labels, inputs, preds, check_until_topk=1, return_mistakes=False):
        c2c, c2i, i2c, i2i = 0, 0, 0, 0
        for lab, inp, pred in zip(labels, inputs, preds):
            for l, i_, p in zip(lab.split(), inp.split(), pred.split()):
                correct = (l == i_)
                predicted_correct = (p == l)
                if correct and predicted_correct:     c2c += 1
                elif correct and not predicted_correct: c2i += 1
                elif not correct and predicted_correct: i2c += 1
                else:                                   i2i += 1
        return c2c, c2i, i2c, i2i

    class _NeuSpellBase:
        """Minimal stand-in for neuspell.corrector.Corrector."""
        def __init__(self, **kwargs):
            self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            self.model, self.vocab = None, None
            self.ckpt_path, self.vocab_path = None, None

        def is_model_ready(self):
            assert self.model is not None and self.vocab is not None, \
                "call from_pretrained() or from_huggingface() first"

        def correct(self, x: str) -> str:
            return self.correct_strings([x])[0]

        correct_string = correct

        def correct_from_file(self, src: str, dest: str = "./clean_version.txt") -> None:
            self.is_model_ready()
            lines = [l.strip() for l in open(src, encoding="utf-8")]
            cleaned = self.correct_strings(lines)
            with open(dest, "w", encoding="utf-8") as f:
                f.write("\n".join(cleaned) + "\n")

        def load_output_vocab(self, vocab_path: str) -> None:
            self.vocab = load_vocab_dict(vocab_path)

        @property
        def get_num_params(self):
            self.is_model_ready()
            return get_model_nparams(self.model)

        def quantize_model(self, print_stats: bool = False):
            self.is_model_ready()
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )


# ---------------------------------------------------------------------------
# Default model identifier
# ---------------------------------------------------------------------------
DEFAULT_BIOCLINICAL_MODERNBERT = "thomas-sounack/BioClinical-ModernBERT-base"
DEFAULT_MAX_SEQ_LEN = 3072   # ModernBERT supports up to 8192; cap at 512 for speed


# ===========================================================================
# Tokenisation helpers
# ===========================================================================

def _tokenize_batch_for_modernbert(
    batch_sentences: List[str],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = DEFAULT_MAX_SEQ_LEN,
) -> Tuple[List[str], Dict[str, torch.Tensor], List[List[Optional[int]]]]:
    """
    Tokenise a batch of space-separated sentences for any fast HuggingFace
    tokenizer (BPE, WordPiece, Unigram …) using the ``word_ids()`` API.

    This avoids the ``##``-prefix hack in the original NeuSpell codebase, which
    was tied to WordPiece / BERT-style tokenisers and would silently break with
    BPE tokenisers such as those used by ModernBERT.

    Parameters
    ----------
    batch_sentences:
        Raw sentences, e.g. ``["Pt c/o worsning dyspenea", ...]``
    tokenizer:
        A *fast* HuggingFace tokenizer (``PreTrainedTokenizerFast``).
    max_length:
        Hard cap on subword sequence length.

    Returns
    -------
    batch_sentences_clean : List[str]
        Sentences after the tokenizer's basic normalisation (lowercasing etc.),
        reconstructed from the tokenizer's own token list so that word counts
        match ``batch_word_ids``.
    batch_bert_dict : Dict[str, Tensor]
        ``input_ids`` and ``attention_mask`` tensors, shape ``(BS, max_len)``.
        ``token_type_ids`` is omitted (ModernBERT does not use it).
    batch_word_ids : List[List[Optional[int]]]
        For each sentence, a list of length ``max_len`` mapping every subword
        token position to its original word index (``None`` for special tokens
        and padding).
    """
    if not tokenizer.is_fast:
        raise TypeError(
            "BioClinicalModernBertChecker requires a fast HuggingFace tokenizer "
            "(PreTrainedTokenizerFast). The supplied tokenizer is slow and does "
            "not expose word_ids()."
        )

    # Split each sentence into words; keep consistent with NeuSpell convention
    batch_word_lists: List[List[str]] = [s.split() for s in batch_sentences]

    # Batch-encode with is_split_into_words=True so each entry in
    # word_lists is treated as a single word before sub-word splitting.
    encoding = tokenizer(
        batch_word_lists,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=False,  # ModernBERT has no segment IDs
    )

    batch_word_ids: List[List[Optional[int]]] = [
        encoding.word_ids(batch_index=i) for i in range(len(batch_sentences))
    ]

    # Reconstruct the "clean" sentence from what the tokenizer actually sees
    # (handles any normalisation the tokenizer performs).  We do this by
    # collecting the unique non-None word ids and using the original word list.
    batch_sentences_clean: List[str] = []
    for i, wids in enumerate(batch_word_ids):
        seen = []
        for w in wids:
            if w is not None and (not seen or seen[-1] != w):
                seen.append(w)
        # Reconstruct from original words (the tokenizer may have altered them)
        batch_sentences_clean.append(" ".join(batch_word_lists[i][w] for w in seen))

    batch_bert_dict = {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
    }

    return batch_sentences_clean, batch_bert_dict, batch_word_ids


def _filter_length_mismatch(
    batch_original: List[str],
    batch_noisy: List[str],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = DEFAULT_MAX_SEQ_LEN,
) -> Tuple[
    List[str], List[str],
    Dict[str, torch.Tensor],
    List[List[Optional[int]]],
]:
    """
    Equivalent of NeuSpell's ``bert_tokenize_for_valid_examples``, but
    tokenizer-agnostic.  Filters out sentence pairs where the number of
    *words* seen by the tokenizer differs between original and noisy versions
    (this can happen when the tokenizer's basic normalisation merges tokens).

    Returns the filtered originals/noisy sentences, the encoded batch dict,
    and the word_ids lists for the noisy sentences (which feed the encoder).
    """
    # Tokenise both sides to discover their effective word counts
    orig_clean, _, orig_wids = _tokenize_batch_for_modernbert(batch_original, tokenizer, max_length)
    noisy_clean, batch_bert_dict, noisy_wids = _tokenize_batch_for_modernbert(batch_noisy, tokenizer, max_length)

    def _word_count(wids):
        seen = set()
        for w in wids:
            if w is not None:
                seen.add(w)
        return len(seen)

    valid_idx = [
        i for i, (o, n) in enumerate(zip(orig_wids, noisy_wids))
        if _word_count(o) == _word_count(n)
    ]

    if not valid_idx:
        return [], [], {}, []

    # Re-encode only the valid subset to get a clean batch dict
    valid_noisy = [batch_noisy[i] for i in valid_idx]
    valid_orig = [batch_original[i] for i in valid_idx]
    orig_clean_f, _, _ = _tokenize_batch_for_modernbert(valid_orig, tokenizer, max_length)
    noisy_clean_f, batch_bert_dict_f, noisy_wids_f = _tokenize_batch_for_modernbert(valid_noisy, tokenizer, max_length)

    return orig_clean_f, noisy_clean_f, batch_bert_dict_f, noisy_wids_f


# ===========================================================================
# Neural model
# ===========================================================================

class SubwordModernBert(nn.Module):
    """
    Subword-level encoder → word-level token classifier.

    Architecture
    ------------
    Input tokens  →  AutoModel (ModernBERT)  →  subword hidden states
                  →  mean-pool per original word (via word_ids)
                  →  Dropout(0.2)
                  →  Linear(hidden_size, |vocab|)
                  →  CrossEntropyLoss

    This is architecturally identical to NeuSpell's ``SubwordBert`` but:
    * Uses ``word_ids`` lists for subword merging instead of ``batch_splits``.
    * Works with any HuggingFace encoder (BPE or WordPiece).
    * Does not hard-code ``token_type_ids``.

    Parameters
    ----------
    padding_idx : int
        Vocab index of the pad token; excluded from the loss.
    output_dim : int
        Target vocabulary size (number of unique output word types).
    pretrained_model_name_or_path : str
        HuggingFace model identifier.
    freeze_encoder : bool
        If True, encoder weights are frozen (useful for very small datasets).
    dropout : float
        Dropout rate applied to the pooled word encodings.
    """

    def __init__(
        self,
        padding_idx: int,
        output_dim: int,
        pretrained_model_name_or_path: str = DEFAULT_BIOCLINICAL_MODERNBERT,
        freeze_encoder: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.hidden_size: int = self.encoder.config.hidden_size
        self._uses_token_type_ids: bool = getattr(
            self.encoder.config, "type_vocab_size", 0
        ) > 1

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        assert output_dim > 0, "output_dim must be > 0"
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.hidden_size, output_dim)
        self.criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=padding_idx)

    # ------------------------------------------------------------------
    # Subword → word pooling
    # ------------------------------------------------------------------

    def _merge_subwords(
        self,
        hidden: torch.Tensor,
        word_ids: List[Optional[int]],
        mode: str = "mean",
    ) -> torch.Tensor:
        """
        Pool subword token representations into one vector per original word.

        Parameters
        ----------
        hidden : Tensor of shape ``(seq_len, hidden_size)``
        word_ids : list of length ``seq_len``
            Maps each position to an original word index, or ``None`` for
            special / padding tokens.
        mode : ``"mean"`` | ``"max"`` | ``"first"``

        Returns
        -------
        Tensor of shape ``(n_words, hidden_size)``
        """
        # Collect the set of word indices that appear (preserving order)
        seen: List[int] = []
        seen_set: set = set()
        for w in word_ids:
            if w is not None and w not in seen_set:
                seen.append(w)
                seen_set.add(w)

        if not seen:
            return hidden.new_zeros(0, self.hidden_size)

        word_vecs: List[torch.Tensor] = []
        for w in seen:
            positions = [i for i, wid in enumerate(word_ids) if wid == w]
            subword_vecs = hidden[positions]          # (n_sub, hidden_size)
            if mode == "mean":
                word_vecs.append(subword_vecs.mean(dim=0))
            elif mode == "max":
                word_vecs.append(subword_vecs.max(dim=0).values)
            elif mode == "first":
                word_vecs.append(subword_vecs[0])
            else:
                raise ValueError(f"Unknown pooling mode: {mode!r}")

        return torch.stack(word_vecs, dim=0)          # (n_words, hidden_size)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        batch_bert_dict: Dict[str, torch.Tensor],
        batch_word_ids: List[List[Optional[int]]],
        targets: Optional[torch.Tensor] = None,
        topk: int = 1,
    ):
        """
        Parameters
        ----------
        batch_bert_dict:
            ``input_ids`` and ``attention_mask`` (and optionally
            ``token_type_ids``), each of shape ``(BS, max_seq_len)``.
        batch_word_ids:
            Per-sentence word_ids lists from the tokenizer.
        targets:
            Ground-truth word indices, shape ``(BS, max_n_words)``.
            Provide during training; omit at inference.
        topk:
            Number of top predictions to return (inference only).

        Returns
        -------
        During training (``self.training == True``):
            ``loss`` — scalar Tensor
        During inference:
            ``(loss_np, predictions_np)`` — (float, ndarray)
        """
        batch_size = len(batch_word_ids)

        # ---- Encoder ----
        encoder_kwargs = {k: v for k, v in batch_bert_dict.items()}
        if not self._uses_token_type_ids:
            encoder_kwargs.pop("token_type_ids", None)

        hidden_states = self.encoder(**encoder_kwargs, return_dict=False)[0]
        # hidden_states: (BS, max_seq_len, hidden_size)

        # ---- Subword → word pooling ----
        word_encodings: List[torch.Tensor] = [
            self._merge_subwords(hidden_states[i], batch_word_ids[i], mode="mean")
            for i in range(batch_size)
        ]
        # Pad to (BS, max_n_words, hidden_size)
        word_encodings_padded = pad_sequence(
            word_encodings, batch_first=True, padding_value=0.0
        )                                               # (BS, max_nwords, hidden)

        # ---- Classifier head ----
        logits = self.classifier(self.dropout(word_encodings_padded))
        # logits: (BS, max_nwords, output_dim)

        # ---- Loss ----
        loss = None
        if targets is not None:
            logits_t = logits.permute(0, 2, 1)         # (BS, output_dim, max_nwords)
            loss = self.criterion(logits_t, targets)

        # ---- Inference predictions ----
        if not self.training:
            probs = F.softmax(logits, dim=-1)
            if topk == 1:
                top_inds = torch.argmax(probs, dim=-1)  # (BS, max_nwords)
            else:
                _, top_inds = torch.topk(probs, topk, dim=-1)   # (BS, max_nwords, topk)

            loss_val = loss.cpu().detach().numpy() if loss is not None else 0.0
            return loss_val, top_inds.cpu().detach().numpy()

        return loss


# ===========================================================================
# Checker class
# ===========================================================================

class BioClinicalModernBertChecker(_NeuSpellBase):
    """
    Drop-in replacement for ``neuspell.BertChecker`` using BioClinical
    ModernBERT as the encoder.

    Usage — training a new model from scratch
    -----------------------------------------
    ::

        from neuspell.seq_modeling.helpers import get_tokens, load_data
        from bioclinical_modernbert_checker import BioClinicalModernBertChecker

        train_data = load_data("data/", "train_clean.txt", "train_noisy.txt")
        vocab = get_tokens([s for s, _ in train_data], keep_simple=True,
                           min_max_freq=(1, float("inf")), topk=100_000)

        checker = BioClinicalModernBertChecker(device="cuda")
        checker.from_huggingface(vocab=vocab)
        checker.finetune(clean_file="train_clean.txt",
                         corrupt_file="train_noisy.txt",
                         data_dir="data/")

    Usage — loading a saved checkpoint
    ------------------------------------
    ::

        checker = BioClinicalModernBertChecker(device="cpu")
        checker.from_pretrained(ckpt_path="checkpoints/my_model")
        checker.correct("Pt c/o worsning dyspenea and fiver")
        # → "Pt c/o worsening dyspnea and fever"
    """

    # Default HuggingFace model used for the encoder
    DEFAULT_ENCODER = DEFAULT_BIOCLINICAL_MODERNBERT

    def __init__(self, **kwargs) -> None:
        # Bypass NeuSpell's name-resolution logic when neuspell is available,
        # since this checker is not in its DEFAULT_CHECKERNAME_TO_NAME_MAPPING.
        if _NEUSPELL_AVAILABLE:
            # Pass a dummy name to prevent KeyError in the parent __init__
            kwargs.setdefault("name", "_bioclinical_modernbert")
            try:
                super().__init__(**kwargs)
            except ModuleNotFoundError:
                # Fallback: init only the fields we need
                self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
                self.device = "cuda" if self.device == "gpu" else self.device
                self.model = None
                self.vocab = None
                self.ckpt_path = None
                self.vocab_path = None
        else:
            super().__init__(**kwargs)

        self.encoder_name: str = kwargs.get("encoder_name", self.DEFAULT_ENCODER)
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None
        self.max_seq_len: int = kwargs.get("max_seq_len", DEFAULT_MAX_SEQ_LEN)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_tokenizer(self) -> None:
        """Load tokenizer lazily (or if encoder_name changed)."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
            if not self.tokenizer.is_fast:
                raise RuntimeError(
                    f"The tokenizer for {self.encoder_name!r} is not a fast "
                    "tokenizer. BioClinicalModernBertChecker requires a fast "
                    "tokenizer for word_ids() support."
                )

    def _batch_size(self) -> int:
        return 4 if self.device == "cpu" else 16

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, ckpt_path: str) -> None:
        """Load a ``pytorch_model.bin`` checkpoint from *ckpt_path*."""
        print("Initialising SubwordModernBert …")
        self._ensure_tokenizer()
        pad_idx = self.vocab["token2idx"][self.vocab["pad_token"]]
        output_dim = len(self.vocab["token2idx"])   # must include <pad> and <unk>

        self.model = SubwordModernBert(
            padding_idx=pad_idx,
            output_dim=output_dim,
            pretrained_model_name_or_path=self.encoder_name,
        )

        model_bin = os.path.join(ckpt_path, "pytorch_model.bin")
        if not os.path.isfile(model_bin):
            raise FileNotFoundError(
                f"No checkpoint found at {model_bin}. "
                "Train with .finetune() or supply a valid ckpt_path."
            )
        map_location = (
            (lambda storage, loc: storage.cuda())
            if torch.cuda.is_available() and self.device != "cpu"
            else "cpu"
        )
        print(f"Loading weights from {model_bin}")
        state = torch.load(model_bin, map_location=map_location)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        print(f"Model parameters: {get_model_nparams(self.model):,}")

    def from_huggingface(
        self,
        vocab: Union[Dict, str],
        encoder_name: Optional[str] = None,
        freeze_encoder: bool = False,
    ) -> "BioClinicalModernBertChecker":
        """
        Initialise a *new* model (no pre-saved weights) ready for fine-tuning.

        Parameters
        ----------
        vocab : dict or str
            Output vocabulary dict (keys: ``token2idx``, ``idx2token``,
            ``token_freq``, ``pad_token``, ``unk_token``), or a path to a
            pickled vocab file.
        encoder_name : str, optional
            Override the default BioClinical ModernBERT model identifier.
        freeze_encoder : bool
            Freeze encoder weights (useful for very small clinical datasets).
        """
        if encoder_name:
            self.encoder_name = encoder_name

        if isinstance(vocab, str) and os.path.exists(vocab):
            self.vocab_path = vocab
            print(f"Loading vocab from {vocab}")
            self.vocab = load_vocab_dict(vocab)
        elif isinstance(vocab, dict):
            self.vocab = vocab
        else:
            raise ValueError(f"Unknown vocab type or path not found: {type(vocab)}")

        self._ensure_tokenizer()
        pad_idx = self.vocab["token2idx"][self.vocab["pad_token"]]
        output_dim = len(self.vocab["token2idx"])   # must include <pad> and <unk>

        print(
            f"Initialising SubwordModernBert\n"
            f"  encoder  : {self.encoder_name}\n"
            f"  vocab sz : {output_dim:,}\n"
            f"  device   : {self.device}"
        )
        self.model = SubwordModernBert(
            padding_idx=pad_idx,
            output_dim=output_dim,
            pretrained_model_name_or_path=self.encoder_name,
            freeze_encoder=freeze_encoder,
        )
        self.model.to(self.device)
        print(f"Model parameters: {get_model_nparams(self.model):,}")
        return self

    def from_pretrained(
        self,
        ckpt_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        encoder_name: Optional[str] = None,
    ) -> "BioClinicalModernBertChecker":
        """
        Load a fine-tuned checkpoint.

        The checkpoint directory must contain:
        - ``pytorch_model.bin``   — model weights
        - ``vocab.pkl``           — output vocabulary
        - ``encoder_name.txt``    — (optional) encoder identifier
        """
        if encoder_name:
            self.encoder_name = encoder_name

        if ckpt_path is None:
            raise ValueError("ckpt_path is required for from_pretrained()")
        self.ckpt_path = ckpt_path

        # Allow overriding the encoder name from the checkpoint metadata
        enc_name_file = os.path.join(ckpt_path, "encoder_name.txt")
        if os.path.isfile(enc_name_file) and not encoder_name:
            saved = Path(enc_name_file).read_text().strip()
            if saved:
                self.encoder_name = saved
                print(f"Encoder name loaded from checkpoint: {self.encoder_name}")

        self.vocab_path = vocab_path or os.path.join(ckpt_path, "vocab.pkl")
        self.load_output_vocab(self.vocab_path)
        self.load_model(ckpt_path)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def correct_strings(
        self,
        mystrings: List[str],
        return_all: bool = False,
    ) -> Union[List[str], Tuple[List[str], List[str]]]:
        """
        Correct a list of noisy sentences.

        Parameters
        ----------
        mystrings :
            Input sentences (space-tokenised).
        return_all :
            If True, return ``(original_normalised, corrected)`` tuple.

        Returns
        -------
        List[str]  (or tuple of two List[str] if return_all)
        """
        self.is_model_ready()
        self._ensure_tokenizer()

        batch_size = self._batch_size()
        all_predictions: List[str] = []
        all_originals: List[str] = []

        data = [(s, s) for s in mystrings]
        for batch_labels, batch_sentences in batch_iter(data, batch_size=batch_size, shuffle=False):
            orig, noisy, bert_dict, word_ids = _filter_length_mismatch(
                batch_labels, batch_sentences, self.tokenizer, self.max_seq_len
            )
            if not orig:
                continue

            bert_dict = {k: v.to(self.device) for k, v in bert_dict.items()}
            label_ids, lengths = labelize(orig, self.vocab)
            label_ids = label_ids.to(self.device)

            self.model.eval()
            with torch.no_grad():
                _, preds = self.model(bert_dict, word_ids, targets=label_ids, topk=1)

            lengths_np = lengths.cpu().numpy()
            decoded = untokenize_without_unks(preds, lengths_np, self.vocab, noisy)
            all_predictions.extend(decoded)
            all_originals.extend(orig)

        if return_all:
            return all_originals, all_predictions
        return all_predictions

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        clean_file: str,
        corrupt_file: str,
        data_dir: str = "",
    ) -> None:
        """
        Evaluate on a clean/corrupt file pair and print accuracy metrics.
        """
        self.is_model_ready()
        self._ensure_tokenizer()

        base = data_dir if data_dir else "."
        test_data = load_data(base, clean_file, corrupt_file)
        batch_size = self._batch_size()

        c2c = c2i = i2c = i2i = 0
        total_loss = 0.0
        n_batches = 0

        self.model.eval()
        self.model.to(self.device)

        for batch_labels, batch_sentences in tqdm(
            batch_iter(test_data, batch_size=batch_size, shuffle=False),
            desc="Evaluating",
            total=int(np.ceil(len(test_data) / batch_size)),
        ):
            orig, noisy, bert_dict, word_ids = _filter_length_mismatch(
                batch_labels, batch_sentences, self.tokenizer, self.max_seq_len
            )
            if not orig:
                continue

            bert_dict = {k: v.to(self.device) for k, v in bert_dict.items()}
            label_ids, lengths = labelize(orig, self.vocab)
            label_ids = label_ids.to(self.device)

            with torch.no_grad():
                loss_val, preds = self.model(bert_dict, word_ids, targets=label_ids, topk=1)

            total_loss += float(loss_val)
            n_batches += 1

            lengths_np = lengths.cpu().numpy()
            decoded = untokenize_without_unks(preds, lengths_np, self.vocab, noisy)
            bc2c, bc2i, bi2c, bi2i = get_metrics(
                orig, noisy, decoded, check_until_topk=1, return_mistakes=False
            )
            c2c += bc2c; c2i += bc2i; i2c += bi2c; i2i += bi2i

        total = c2c + c2i + i2c + i2i
        print("\n" + "=" * 55)
        print(f"Evaluation results")
        print(f"  data size           : {len(test_data):,}")
        print(f"  total tokens        : {total:,}")
        print(f"  avg loss            : {total_loss / max(n_batches, 1):.4f}")
        print(f"  corr→corr  : {c2c:,}   corr→incorr : {c2i:,}")
        print(f"  incorr→corr: {i2c:,}   incorr→incorr: {i2i:,}")
        if total > 0:
            print(f"  accuracy            : {(c2c + i2c) / total:.4%}")
        if (i2c + i2i) > 0:
            print(f"  word correction rate: {i2c / (i2c + i2i):.4%}")
        print("=" * 55)

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def finetune(
        self,
        clean_file: str,
        corrupt_file: str,
        data_dir: str = "",
        validation_split: float = 0.2,
        n_epochs: int = 3,
        learning_rate: float = 2e-5,
        train_batch_size: int = 16,
        valid_batch_size: int = 32,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.06,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        save_dir: Optional[str] = None,
        new_vocab_list: Optional[List] = None,
    ) -> str:
        """
        Fine-tune the model on a custom parallel corpus.

        Parameters
        ----------
        clean_file, corrupt_file :
            Line-aligned plain-text files (one sentence per line).
        data_dir :
            Directory containing the above files.
        validation_split :
            Fraction of data held out for validation.
        n_epochs :
            Number of full training epochs.
        learning_rate :
            Peak AdamW learning rate.
        train_batch_size, valid_batch_size :
            Batch sizes (before gradient accumulation).
        gradient_accumulation_steps :
            Accumulate gradients over this many steps before a parameter update
            (increases effective batch size without extra memory).
        warmup_ratio :
            Fraction of total steps used for linear LR warm-up.
        max_grad_norm :
            Gradient clipping threshold.
        use_amp :
            Enable automatic mixed precision on CUDA.
        save_dir :
            Where to save checkpoints.  Defaults to
            ``<data_dir>/new_models/bioclinical-modernbert``.
        new_vocab_list :
            Not supported (output vocab is fixed at model init time).

        Returns
        -------
        str — path to the saved checkpoint directory.
        """
        if new_vocab_list:
            raise NotImplementedError(
                "Modifying the output vocabulary after model initialisation is "
                "not currently supported."
            )

        self.is_model_ready()
        self._ensure_tokenizer()

        # ---- Data ----
        base = data_dir if data_dir else "."
        all_data = load_data(base, clean_file, corrupt_file)
        train_data, valid_data = train_validation_split(
            all_data, 1.0 - validation_split, seed=42
        )
        print(f"Train: {len(train_data):,}  Valid: {len(valid_data):,}")

        # ---- Checkpoint directory ----
        if save_dir is None:
            save_dir = os.path.join(
                self.ckpt_path if self.ckpt_path else base,
                "new_models",
                "bioclinical-modernbert",
            )
        # Avoid overwriting existing checkpoints
        final_save_dir = save_dir
        run = 1
        while os.path.exists(final_save_dir):
            final_save_dir = f"{save_dir}-run{run}"
            run += 1
        os.makedirs(final_save_dir, exist_ok=True)
        print(f"Checkpoints → {final_save_dir}")

        # ---- AMP ----
        amp_enabled = use_amp and self.device != "cpu" and torch.cuda.is_available()
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        # ---- Optimiser and scheduler ----
        no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
        param_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)

        steps_per_epoch = int(np.ceil(len(train_data) / train_batch_size))
        t_total = steps_per_epoch // gradient_accumulation_steps * n_epochs
        n_warmup = max(1, int(warmup_ratio * t_total))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=n_warmup, num_training_steps=t_total
        )

        self.model.to(self.device)

        best_valid_acc = -1.0
        best_epoch = -1
        log_path = os.path.join(final_save_dir, "training_log.txt")
        vocab = self.vocab

        with open(log_path, "w", encoding="utf-8") as log_f:
            log_f.write(
                f"encoder : {self.encoder_name}\n"
                f"train_sz: {len(train_data)}\n"
                f"valid_sz: {len(valid_data)}\n"
                f"lr      : {learning_rate}\n"
                f"epochs  : {n_epochs}\n"
                f"amp     : {amp_enabled}\n\n"
            )
            log_f.flush()

            for epoch in range(1, n_epochs + 1):
                print(f"\n{'─'*55}\nEpoch {epoch}/{n_epochs}")
                log_f.write(f"\n=== Epoch {epoch} ===\n")

                # ---- Training ----
                self.model.train()
                train_loss = 0.0
                train_acc = train_acc_n = 0
                optimizer.zero_grad()

                t_iter = tqdm(
                    batch_iter(train_data, batch_size=train_batch_size, shuffle=True),
                    desc="  train",
                    total=steps_per_epoch,
                )
                for step, (batch_labels, batch_sentences) in enumerate(t_iter, 1):
                    orig, noisy, bert_dict, word_ids = _filter_length_mismatch(
                        batch_labels, batch_sentences, self.tokenizer, self.max_seq_len
                    )
                    if not orig:
                        continue

                    bert_dict = {k: v.to(self.device) for k, v in bert_dict.items()}
                    label_ids, lengths = labelize(orig, vocab)
                    label_ids = label_ids.to(self.device)

                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        loss = self.model(bert_dict, word_ids, targets=label_ids)

                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps

                    scaler.scale(loss).backward()

                    if step % gradient_accumulation_steps == 0 or step >= steps_per_epoch:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()

                    train_loss += loss.item() * gradient_accumulation_steps

                    # Periodic accuracy snapshot (cheap, every 500 steps)
                    if step % 500 == 0:
                        self.model.eval()
                        with torch.no_grad():
                            _, preds_np = self.model(
                                bert_dict, word_ids, targets=label_ids, topk=1
                            )
                        self.model.train()
                        ncorr, ntotal = batch_accuracy_func(
                            preds_np,
                            label_ids.cpu().numpy(),
                            lengths.cpu().numpy(),
                        )
                        train_acc += ncorr / max(ntotal, 1)
                        train_acc_n += 1

                    t_iter.set_postfix(loss=f"{train_loss / step:.4f}")

                avg_train_loss = train_loss / max(steps_per_epoch, 1)
                avg_train_acc = train_acc / max(train_acc_n, 1)
                print(
                    f"  train loss: {avg_train_loss:.4f}  "
                    f"  train acc (spot): {avg_train_acc:.4f}"
                )
                log_f.write(
                    f"  train_loss={avg_train_loss:.4f}  train_acc_spot={avg_train_acc:.4f}\n"
                )
                log_f.flush()

                # ---- Validation ----
                self.model.eval()
                valid_loss = valid_acc = valid_acc_n = 0.0
                v_steps = int(np.ceil(len(valid_data) / valid_batch_size))

                for batch_labels, batch_sentences in tqdm(
                    batch_iter(valid_data, batch_size=valid_batch_size, shuffle=False),
                    desc="  valid",
                    total=v_steps,
                ):
                    orig, noisy, bert_dict, word_ids = _filter_length_mismatch(
                        batch_labels, batch_sentences, self.tokenizer, self.max_seq_len
                    )
                    if not orig:
                        continue

                    bert_dict = {k: v.to(self.device) for k, v in bert_dict.items()}
                    label_ids, lengths = labelize(orig, vocab)
                    label_ids = label_ids.to(self.device)

                    with torch.no_grad():
                        with torch.amp.autocast("cuda", enabled=amp_enabled):
                            loss_val, preds_np = self.model(
                                bert_dict, word_ids, targets=label_ids, topk=1
                            )

                    valid_loss += float(loss_val)
                    ncorr, ntotal = batch_accuracy_func(
                        preds_np,
                        label_ids.cpu().numpy(),
                        lengths.cpu().numpy(),
                    )
                    valid_acc += ncorr / max(ntotal, 1)
                    valid_acc_n += 1

                avg_valid_loss = valid_loss / max(v_steps, 1)
                avg_valid_acc = valid_acc / max(valid_acc_n, 1)
                print(
                    f"  valid loss: {avg_valid_loss:.4f}  "
                    f"  valid acc : {avg_valid_acc:.4f}"
                )
                log_f.write(
                    f"  valid_loss={avg_valid_loss:.4f}  valid_acc={avg_valid_acc:.4f}\n"
                )
                log_f.flush()

                # Save best checkpoint
                if avg_valid_acc >= best_valid_acc:
                    best_valid_acc = avg_valid_acc
                    best_epoch = epoch
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(final_save_dir, "pytorch_model.bin"),
                    )
                    save_vocab_dict(os.path.join(final_save_dir, "vocab.pkl"), vocab)
                    Path(os.path.join(final_save_dir, "encoder_name.txt")).write_text(
                        self.encoder_name
                    )
                    print(f"  ✓ Checkpoint saved (best valid_acc = {best_valid_acc:.4f})")
                    log_f.write("  [checkpoint saved]\n")

            log_f.write(
                f"\nTraining complete. Best epoch: {best_epoch}  "
                f"Best valid_acc: {best_valid_acc:.4f}\n"
            )

        print(
            f"\nTraining complete.\n"
            f"  Best epoch     : {best_epoch}\n"
            f"  Best valid acc : {best_valid_acc:.4f}\n"
            f"  Checkpoint dir : {final_save_dir}"
        )
        self.ckpt_path = final_save_dir
        return final_save_dir


# ===========================================================================
# Convenience factory
# ===========================================================================

def build_vocab_from_files(
    clean_file: str,
    corrupt_file: Optional[str] = None,
    data_dir: str = ".",
    min_freq: int = 1,
    topk: int = 100_000,
    keep_simple: bool = True,
) -> dict:
    """
    Build an output vocabulary from a parallel corpus.

    Wrapper around NeuSpell's ``get_tokens`` that accepts file paths directly.

    Parameters
    ----------
    clean_file :
        Path (relative to *data_dir*) of the clean-text file (target labels).
    corrupt_file :
        Optional path to noisy file; if given, its tokens are also included
        so that the vocab covers the model's input domain.
    data_dir :
        Base directory.
    min_freq :
        Minimum token frequency.
    topk :
        Maximum vocabulary size (most frequent tokens retained).
    keep_simple :
        If True, filter out tokens containing digits or non-ASCII characters.
        **Set to False for clinical text** to preserve e.g. "IL-6", "HbA1c".

    Returns
    -------
    dict — NeuSpell-format vocabulary.
    """
    def _read(path):
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    texts = _read(os.path.join(data_dir, clean_file))
    if corrupt_file:
        texts += _read(os.path.join(data_dir, corrupt_file))

    return get_tokens(
        texts,
        keep_simple=keep_simple,
        min_max_freq=(min_freq, float("inf")),
        topk=topk,
    )


# ===========================================================================
# Quick self-test
# ===========================================================================

if __name__ == "__main__":
    """
    Minimal smoke-test (no GPU, no saved weights needed).

    Runs a single forward pass with a randomly initialised model to verify
    the tokenisation pipeline and model shapes are correct.
    """
    import tempfile

    print("=== BioClinicalModernBertChecker smoke test ===\n")

    # --- Build a tiny toy vocab ---
    toy_sentences = [
        "the patient has dyspnea and fever",
        "medication was administered intravenously",
        "blood pressure is elevated this morning",
        "cardiac catheterisation was performed",
    ]
    vocab = get_tokens(toy_sentences, keep_simple=False, min_max_freq=(1, float("inf")))
    print(f"Vocab size: {len(vocab['token2idx'])}")

    # --- Initialise checker ---
    checker = BioClinicalModernBertChecker(device="cpu")
    checker.from_huggingface(vocab=vocab)

    # --- Forward pass ---
    test_inputs = [
        "the patint has dyspenea and fiver",
        "medicaton was adminstered intravenusly",
    ]
    print("\nInput sentences:")
    for s in test_inputs:
        print(f"  {s!r}")

    predictions = checker.correct_strings(test_inputs)
    print("\nPredicted corrections (untrained — random output expected):")
    for orig, pred in zip(test_inputs, predictions):
        print(f"  {orig!r}")
        print(f"  → {pred!r}")

    # --- Save and reload ---
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write tiny train/valid files
        clean = os.path.join(tmpdir, "clean.txt")
        noisy = os.path.join(tmpdir, "noisy.txt")
        with open(clean, "w", encoding="utf-8") as f:
            f.write("\n".join(toy_sentences))
        with open(noisy, "w", encoding="utf-8") as f:
            f.write("\n".join(toy_sentences))   # identical (no errors) for smoke test

        print("\nRunning 1-epoch fine-tune smoke test (CPU, tiny data) …")
        ckpt = checker.finetune(
            clean_file="clean.txt",
            corrupt_file="noisy.txt",
            data_dir=tmpdir,
            n_epochs=1,
            train_batch_size=2,
            valid_batch_size=2,
            gradient_accumulation_steps=1,
            use_amp=False,
            validation_split=0.25,
        )

        print(f"\nReloading from checkpoint: {ckpt}")
        checker2 = BioClinicalModernBertChecker(device="cpu")
        checker2.from_pretrained(ckpt_path=ckpt)
        result = checker2.correct("the patint has dyspenea")
        print(f"Post-reload correction: {result!r}")

    print("\n✓ Smoke test passed.")
