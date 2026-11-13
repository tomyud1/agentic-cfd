"""Minimal OpenFOAM dictionary parser: flattens a dictionary file to {path: value} settings.

Used by the report contract to list every setting a case actually uses, so each one can be
required to appear in the report's Decisions section. Not a full OpenFOAM parser: list values
(vertices, blocks, boundary lists, nonuniform fields) are kept as raw text under their key.
"""
import re
from pathlib import Path

_SPECIAL = set("{};")


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def tokenize(text):
    text = _strip_comments(text)
    i, n, toks = 0, len(text), []
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c in _SPECIAL or c in "()":
            toks.append(c)
            i += 1
        elif c == '"':
            j = text.index('"', i + 1)
            toks.append(text[i:j + 1])
            i = j + 1
        else:  # word; may contain balanced parentheses, e.g. div(phi,U)
            j, depth = i, 0
            while j < n:
                d = text[j]
                if d.isspace() or d in _SPECIAL:
                    break
                if d == "(":
                    depth += 1
                elif d == ")":
                    if depth == 0:
                        break
                    depth -= 1
                j += 1
            toks.append(text[i:j])
            i = j
    return toks


def _parse_entries(toks, i, end):
    """Parse entries until token `end` ('}' or None). Returns (dict, next index)."""
    out = {}
    while i < len(toks):
        t = toks[i]
        if t == end:
            return out, i + 1
        if t == ";":
            i += 1
            continue
        key = t.strip('"')
        i += 1
        if key.startswith("#"):                           # directive: #include "x", #includeEtc "x"
            if i < len(toks) and toks[i] not in ("{", "}", ";"):
                out[key + " " + toks[i].strip('"')] = ""
                i += 1
            continue
        if i < len(toks) and toks[i] == "{":
            sub, i = _parse_entries(toks, i + 1, "}")
            out[key] = sub
            continue
        vals, depth = [], 0
        while i < len(toks):
            v = toks[i]
            if depth == 0 and v == ";":
                i += 1
                break
            if depth == 0 and v == "}":                   # entry without ';' at end of dict
                break
            if v in ("(", "{"):
                depth += 1
            elif v in (")", "}"):
                depth -= 1
            vals.append(v)
            i += 1
        out[key] = " ".join(vals)
    return out, i


def parse(path):
    toks = tokenize(Path(path).read_text(errors="replace"))
    d, _ = _parse_entries(toks, 0, None)
    return d


def flatten(d, prefix=""):
    flat = {}
    for k, v in d.items():
        p = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten(v, p) if v else {p: "{}"})
        else:
            flat[p] = v
    return flat


def case_settings(case_dir, unparsed=None):
    """All settings in a case's 0/, constant/ and system/ dictionaries, as {'system/fvSolution/...': value}.
    unparsed: a list that receives the files that could not be parsed (their settings are missing)."""
    case_dir = Path(case_dir)
    settings = {}
    for sub in ("0", "constant", "system"):
        base = case_dir / sub
        if not base.is_dir():
            continue
        for f in sorted(base.iterdir()):
            if f.is_file() and not f.name.startswith(".") and f.suffix not in (".orig", ".gz"):
                try:
                    flat = flatten(parse(f))
                except (ValueError, IndexError):
                    if unparsed is not None:
                        unparsed.append(f"{sub}/{f.name}")
                    continue
                for k, v in flat.items():
                    settings[f"{sub}/{f.name}/{k}"] = v
    return settings
