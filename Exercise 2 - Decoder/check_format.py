#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# In current directory with the expected files:
python check_format.py
"""

import os
import sys
import ast
import re
from typing import Any, Dict, List, Tuple

# Exercise 1 vocab (decoder predictions)
VOCAB = {"[BOS]", "I", "play", "tennis", "[EOS]"}
# Exercise 2 vocab order (for probability vectors)
VOCAB_E2 = ["cat", "sleeps", "the", "ok", "what?"]

# ---------- Utils ----------

def is_numeric(x: Any) -> bool:
    return isinstance(x, (int, float))

def check_matrix(mat: Any) -> Tuple[bool, str]:
    """
    Matrix = non-empty list of non-empty lists of numbers; rectangular.
    """
    if not isinstance(mat, list) or len(mat) == 0:
        return False, "Not a non-empty list"
    if not all(isinstance(row, list) and len(row) > 0 for row in mat):
        return False, "All rows must be non-empty lists"
    ncols = len(mat[0])
    if not all(len(row) == ncols for row in mat):
        return False, "Rows have different lengths (not rectangular)"
    for r in mat:
        for el in r:
            if not is_numeric(el):
                return False, f"Non-numeric element found: {repr(el)}"
    return True, ""

def read_assignments_singleline(path: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Reads assignments of the form: NAME = <python-literal> (one line per var).
    Returns (vars_dict, errors)
    """
    errors: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
    except FileNotFoundError:
        return {}, [f"File not found: {path}"]
    except Exception as e:
        return {}, [f"Error opening {path}: {e}"]

    vars_out: Dict[str, Any] = {}
    for ln in lines:
        if not ln or "=" not in ln:
            continue
        # Expect one assignment per line
        name, rhs = ln.split("=", 1)
        name = name.strip()
        rhs = rhs.strip()
        try:
            val = ast.literal_eval(rhs)
        except Exception as e:
            errors.append(f"Could not parse {name}: {e}")
            continue
        vars_out[name] = val

    return vars_out, errors

# ---------- File checks (Exercise 1) ----------

def check_cheating_decoder(path: str) -> List[str]:
    """
    Format checks for cheating-decoder.txt
    """
    EXP_VARS = ["X", "W_Q", "W_K", "W_V", "W_O", "W_U"]
    errors: List[str] = []

    vars_out, parse_errs = read_assignments_singleline(path)
    errors.extend(parse_errs)
    if parse_errs:
        return errors

    # Presence & type checks
    for var in EXP_VARS:
        if var not in vars_out:
            errors.append(f"Missing variable: {var}")
            continue
        ok, msg = check_matrix(vars_out[var])
        if not ok:
            errors.append(f"{var} must be a 2D numeric list (matrix): {msg}")

    if errors:
        return errors

    # Light shape checks implied by the spec
    X = vars_out["X"]
    nrows_X = len(X)
    ncols_X = len(X[0])
    if nrows_X != 5:
        errors.append(f"X must have 5 rows (got {nrows_X}).")

    d_model = ncols_X

    # W_Q, W_K, W_V must have d_model rows
    for var in ["W_Q", "W_K", "W_V"]:
        mat = vars_out[var]
        if len(mat) != d_model:
            errors.append(f"{var} must have {d_model} rows (got {len(mat)}).")

    # W_O must be dv x d_model, where dv = columns of W_V
    W_V = vars_out["W_V"]
    dv = len(W_V[0])
    W_O = vars_out["W_O"]
    if len(W_O) != dv:
        errors.append(f"W_O must have {dv} rows (got {len(W_O)}).")
    if len(W_O[0]) != d_model:
        errors.append(f"W_O must have {d_model} columns (got {len(W_O[0])}).")

    # W_U must be d_model x 5 (5 = |V|)
    W_U = vars_out["W_U"]
    if len(W_U) != d_model:
        errors.append(f"W_U must have {d_model} rows (got {len(W_U)}).")
    ncols_WU = len(W_U[0])
    if ncols_WU != 5:
        errors.append(f"W_U must have 5 columns (got {ncols_WU}).")

    return errors

def check_explanation(path: str) -> List[str]:
    """
    Explanation file must exist and be non-empty text.
    (No parsing beyond presence and non-empty content.)
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return [f"File not found: {path}"]
    except Exception as e:
        return [f"Error opening {path}: {e}"]

    if not isinstance(content, str) or not content.strip():
        return ["Explanation must be non-empty text."]
    return []

def _extract_section(lines: List[str], header: str) -> Tuple[List[str], int]:
    """
    Return the lines in a section starting at 'header' (exclusive),
    stopping right before the next header or EOF. Also returns the
    index where the section starts (header index) or -1 if not found.
    """
    try:
        start_idx = lines.index(header)
    except ValueError:
        return [], -1
    section = []
    for i in range(start_idx + 1, len(lines)):
        if lines[i].startswith("Input: Predicted next token"):
            break
        if lines[i].strip():
            section.append(lines[i].strip())
    return section, start_idx

def check_predictions(path: str) -> List[str]:
    """
    Checks presence & structure of the two required sections and
    ensures each section has exactly 4 lines: '<input> <pred>' with tokens in VOCAB.
    """
    errors: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            # Keep raw lines, but normalize by stripping trailing newlines
            raw_lines = [ln.rstrip("\n") for ln in f.readlines()]
    except FileNotFoundError:
        return [f"File not found: {path}"]
    except Exception as e:
        return [f"Error opening {path}: {e}"]

    # Normalize by stripping outer whitespace but preserve headers exactly
    lines = [ln.strip() for ln in raw_lines]

    header_unmasked = "Input: Predicted next token:"
    header_masked   = "Input: Predicted next token (with mask):"

    sec1, idx1 = _extract_section(lines, header_unmasked)
    sec2, idx2 = _extract_section(lines, header_masked)

    if idx1 == -1:
        errors.append(f"Missing header: '{header_unmasked}'")
    if idx2 == -1:
        errors.append(f"Missing header: '{header_masked}'")
    if errors:
        return errors

    # Expected inputs in order
    expected_inputs = ["[BOS]", "I", "play", "tennis"]

    def validate_section(sec_lines: List[str], label: str):
        if len(sec_lines) != 4:
            errors.append(
                f"{label} must have exactly 4 lines after the header (got {len(sec_lines)})."
            )
            return
        for i, line in enumerate(sec_lines):
            parts = line.split()
            if len(parts) != 2:
                errors.append(
                    f"{label} line {i+1} must have exactly two tokens: '<input> <prediction>'. Got: '{line}'"
                )
                continue
            inp, pred = parts[0], parts[1]
            # Input token must be the expected one for that row
            if inp != expected_inputs[i]:
                errors.append(
                    f"{label} line {i+1}: expected input token '{expected_inputs[i]}', got '{inp}'."
                )
            # Prediction must be a single vocabulary token
            if pred not in VOCAB:
                errors.append(
                    f"{label} line {i+1}: predicted token '{pred}' not in vocabulary {sorted(VOCAB)}."
                )

    validate_section(sec1, "Unmasked section")
    validate_section(sec2, "Masked section")

    return errors

# ---------- File checks (Exercise 2) ----------

def _check_prob_vector(vec: Any, name: str) -> List[str]:
    """
    Probability vector: list of 5 numeric, non-negative values (sum > 0).
    We do not enforce exact normalization here (format-only), but check positivity/sanity.
    """
    errors: List[str] = []
    if not isinstance(vec, list) or len(vec) != 5:
        return [f"{name} must be a list of length 5 (order: {VOCAB_E2})."]
    s = 0.0
    for i, v in enumerate(vec):
        if not is_numeric(v):
            errors.append(f"{name}[{i}] must be numeric (got {type(v)}).")
        elif v < 0:
            errors.append(f"{name}[{i}] must be non-negative (got {v}).")
        else:
            s += float(v)
    if s <= 0:
        errors.append(f"{name} must have a positive sum (got {s}).")
    return errors

def check_tiny_decoder_positional(path: str) -> List[str]:
    """
    Format checks for tiny-decoder-positional.txt (Exercise 2).
    Requires variables:
      - probs1noPE : list[5] of numbers (>=0), order VOCAB_E2
      - probs2noPE : list[5] of numbers (>=0), order VOCAB_E2
      - PE         : 3x2 numeric matrix
      - probs1PE   : list[5] of numbers (>=0), order VOCAB_E2
      - probs2PE   : list[5] of numbers (>=0), order VOCAB_E2
    """
    errors: List[str] = []

    vars_out, parse_errs = read_assignments_singleline(path)
    errors.extend(parse_errs)
    if parse_errs:
        return errors

    needed = ["probs1noPE", "probs2noPE", "PE", "probs1PE", "probs2PE"]
    for k in needed:
        if k not in vars_out:
            errors.append(f"Missing variable: {k}")

    if errors:
        return errors

    # Prob vectors
    for k in ["probs1noPE", "probs2noPE", "probs1PE", "probs2PE"]:
        errors.extend(_check_prob_vector(vars_out[k], k))

    # PE matrix shape and numeric content
    PE = vars_out["PE"]
    ok, msg = check_matrix(PE)
    if not ok:
        errors.append(f"PE must be a 2D numeric list (matrix): {msg}")
    else:
        if len(PE) != 3 or len(PE[0]) != 2:
            errors.append(f"PE must be a 3x2 matrix (got {len(PE)}x{len(PE[0])}).")

    return errors

# ---------- Main ----------

def main(folder: str) -> int:
    # Accept both dash and underscore for predictions/explanation
    files_and_checkers = [
        ("cheating_decoder.txt", check_cheating_decoder),
        ("decoder_explanation.txt", check_explanation),
        ("decoder_predictions.txt", check_predictions),
        ("decoder_predictions.txt", check_predictions),  # alternate name accepted
        # Exercise 2:
        ("tiny_decoder_positional.txt", check_tiny_decoder_positional),
    ]

    any_errors = False
    seen_any = False
    for fname, checker in files_and_checkers:
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            # Only warn about missing explanation/predictions alternates implicitly via presence of the other
            continue
        seen_any = True
        errs = checker(path)
        if not errs:
            print(f"✅ {fname}: OK")
        else:
            any_errors = True
            print(f"❌ {fname}:")
            for e in errs:
                print(f"   - {e}")

    if not seen_any:
        print("❌ No expected files found in the folder.")
        return 1

    if not any_errors:
        print("\nAll files are correctly formatted ✅")
        return 0
    else:
        print("\nFormatting issues detected ❗")
        return 1

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    sys.exit(main(folder))
