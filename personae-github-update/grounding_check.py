"""
Automated grounding verification.

Risk addressed: an LLM-written persona could cite a number that is not actually
true of the cluster (a hallucination). Instead of eyeballing each persona, this
module extracts every number mentioned in a persona's text and checks that each
one corresponds to a real statistic of that segment (a feature mean, the segment
size, or its share of the book) within a rounding tolerance.

Returns a per-persona verdict {passed, n_checked, n_matched, unmatched, rate}.
Because the deterministic generator builds its text *from* the statistics, it
should score 100%; the same check flags fabricated figures the moment generation
is routed through a live LLM (PERSONAE_USE_LLM=1).
"""
import re

# numbers like  $7,682   47.6%   0.95   8,950   3.67   1,013
_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")


def _parse_numbers(text):
    out = []
    for tok in _NUM_RE.findall(str(text)):
        cleaned = tok.replace("$", "").replace(",", "").replace("%", "")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def _truth_values(seg_means, size, total):
    """Every number a grounded persona is allowed to state."""
    vals = {float(size), round(100.0 * size / total, 1)}
    for v in seg_means.values:
        v = float(v)
        vals.add(v)
        vals.add(round(v, 2))
        vals.add(round(v))          # money rounded to whole units
    return vals


def _matches(n, truths, rel_tol=0.03, abs_tol=0.5):
    for t in truths:
        denom = max(abs(t), 1.0)
        if abs(n - t) <= abs_tol or abs(n - t) / denom <= rel_tol:
            return True
    return False


def _strip_feature_names(text, feature_names):
    """Remove feature-name substrings before parsing, so digits that are part
    of a column name (e.g. 'Spending Score (1-100)', 'Q4_2024_spend') are not
    mistaken for cited statistics."""
    import re
    out = str(text)
    for name in feature_names:
        variants = {str(name), str(name).replace("_", " "),
                    re.sub(r"\s*\([^)]*\)", "", str(name)),
                    re.sub(r"\s*\([^)]*\)", "", str(name)).replace("_", " ")}
        for v in variants:
            if v.strip():
                out = re.sub(re.escape(v), " ", out, flags=re.IGNORECASE)
    return out


def verify_persona(persona, seg_means, size, total, feature_names=None):
    truths = _truth_values(seg_means, size, total)
    feature_names = list(feature_names) if feature_names is not None else list(seg_means.index)
    fields = [persona.get("tagline", ""), persona.get("description", ""),
              persona.get("marketing_play", "")]
    fields += [str(v) for _, v in persona.get("key_stats", [])]

    checked, unmatched = [], []
    for text in fields:
        text = _strip_feature_names(text, feature_names)
        for n in _parse_numbers(text):
            checked.append(n)
            if not _matches(n, truths):
                unmatched.append(n)

    n_checked = len(checked)
    n_matched = n_checked - len(unmatched)
    return {
        "passed": len(unmatched) == 0,
        "n_checked": n_checked,
        "n_matched": n_matched,
        "unmatched": sorted(set(unmatched)),
        "rate": round(100.0 * n_matched / n_checked, 1) if n_checked else 100.0,
    }


def verify_all(personas, result):
    """Attach a 'grounding' verdict to each persona; return a summary."""
    means, sizes = result["means"], result["sizes"]
    total = int(sizes.sum())
    feature_names = list(means.columns)
    total_checked = total_matched = 0
    for p in personas:
        seg = p["segment"]
        v = verify_persona(p, means.loc[seg], int(sizes[seg]), total, feature_names)
        p["grounding"] = v
        total_checked += v["n_checked"]
        total_matched += v["n_matched"]
    return {
        "all_passed": all(p["grounding"]["passed"] for p in personas),
        "n_personas": len(personas),
        "n_passed": sum(1 for p in personas if p["grounding"]["passed"]),
        "figures_checked": total_checked,
        "figures_matched": total_matched,
        "rate": round(100.0 * total_matched / total_checked, 1) if total_checked else 100.0,
    }
