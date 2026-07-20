"""
Persona generation — translate a segment's statistical profile into a
named, grounded marketing persona.

Two modes, chosen automatically from the uploaded columns:
  * Credit-card schema  -> hand-tuned archetypes (Cash-Advance Revolver, etc.)
  * ANY other dataset   -> personas built from whichever features actually
                           distinguish each segment (name, description, and
                           key stats all derived from the data itself)

Every claim is grounded in the segment's real per-feature averages; nothing
is invented. Set PERSONAE_USE_LLM=1 (+ ANTHROPIC_API_KEY) to route generation
through an LLM instead (see `_llm_persona`).
"""
import os
import numpy as np

PALETTE = ["#A3B18A", "#C48B9F", "#3A5A40", "#C9A66B", "#7A9E7E", "#B5838D",
           "#606C38", "#DDBEA9"]

# Canonical credit-card columns; used only to decide which mode to use.
CC_FEATURES = {
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE",
}


def is_credit_card_schema(features):
    return len(CC_FEATURES.intersection(set(features))) >= 6


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _zprofile(seg_means, overall_means, overall_std):
    std = overall_std.replace(0, 1e-9)
    return (seg_means - overall_means) / std


def _fmt_money(v):
    return f"${v:,.0f}"


def _fmt_num(v):
    """Format an arbitrary feature value readably (no forced currency)."""
    v = float(v)
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:,.1f}"
    if a == int(a):
        return f"{int(v)}"
    return f"{v:.2f}"


def _title(feat):
    return feat.replace("_", " ").strip().title()


# --------------------------------------------------------------------------
# GENERIC mode — works for any numeric dataset
# --------------------------------------------------------------------------
def _dir_adj(z):
    if z >= 0.8:   return "High"
    if z >= 0.35:  return "Above-Average"
    if z <= -0.8:  return "Low"
    if z <= -0.35: return "Below-Average"
    return "Average"


def _dir_word(z):
    return _dir_adj(z).lower().replace("-", "-")


def _distinctive(z_series):
    """Features sorted by how far this segment sits from the population."""
    return sorted(z_series.items(), key=lambda kv: abs(kv[1]), reverse=True)


def _generic_persona(seg, seg_means, z, size, total):
    ranked = _distinctive(z)                     # [(feature, z), ...]
    strong = [(f, zz) for f, zz in ranked if abs(zz) >= 0.35]
    lead = strong if strong else ranked[:2]

    # ---- name from the top 1-2 distinguishing features ----
    f1, z1 = lead[0]
    if len(lead) > 1 and abs(lead[1][1]) >= 0.5:
        f2, z2 = lead[1]
        name = f"The {_dir_adj(z1)}-{_title(f1).replace(' ', '')} " \
               f"& {_dir_adj(z2)}-{_title(f2).replace(' ', '')} Segment"
        name = f"The {_dir_adj(z1)} {_title(f1)}, {_dir_adj(z2)} {_title(f2)} Group"
        tagline = f"Stands out for {_dir_word(z1)} {_title(f1).lower()} and {_dir_word(z2)} {_title(f2).lower()}."
    else:
        name = f"The {_dir_adj(z1)} {_title(f1)} Segment"
        tagline = f"Defined by {_dir_word(z1)} {_title(f1).lower()}."

    # ---- description grounded in the actual averages ----
    phrases = [f"{_dir_word(zz)} {_title(f).lower()}" for f, zz in lead[:3]]
    if len(phrases) > 1:
        phrase_txt = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
    else:
        phrase_txt = phrases[0]
    facts = "; ".join(f"{_title(f)} {_fmt_num(seg_means[f])}" for f, _ in lead[:3])
    pct = 100 * size / total
    desc = (
        f"This segment is {pct:.1f}% of your customers ({size:,}). "
        f"Relative to the average customer, it stands out for {phrase_txt}. "
        f"Defining averages — {facts}. "
        f"Every figure is this segment's own mean, so the persona is a "
        f"plain-language reading of the cluster, not an assumed profile."
    )

    # ---- key stats = the most distinguishing features, actual values ----
    key_stats = [[_title(f), _fmt_num(seg_means[f])] for f, _ in ranked[:5]]

    # ---- a grounded, data-driven recommendation ----
    play = (
        f"Target this group with messaging built around its defining trait — "
        f"{_dir_word(z1)} {_title(f1).lower()}"
    )
    if len(lead) > 1 and abs(lead[1][1]) >= 0.5:
        play += f" paired with {_dir_word(lead[1][1])} {_title(lead[1][0]).lower()}"
    play += ". Tailor the offer and channel to that profile rather than sending one message to everyone."

    return {
        "segment": int(seg), "name": name, "tagline": tagline,
        "size": int(size), "pct": round(pct, 1),
        "color": PALETTE[int(seg) % len(PALETTE)],
        "description": desc, "marketing_play": play, "key_stats": key_stats,
    }


# --------------------------------------------------------------------------
# CREDIT-CARD mode — hand-tuned archetypes
# --------------------------------------------------------------------------
def _cc_name(z):
    ca = z.get("CASH_ADVANCE", 0); pur = z.get("PURCHASES", 0)
    freq = z.get("PURCHASES_FREQUENCY", 0)
    if ca > 0.8 and pur < 0.3:
        return "The Cash-Advance Revolver", "Carries a high balance, borrows cash, rarely shops."
    if pur > 1.0:
        return "The Premium Power-Spender", "Small in number, huge in spend."
    if freq > 0.5:
        return "The Everyday Installment Shopper", "Frequent, mid-size, installment-leaning spender."
    if freq < -0.3 and pur < -0.2:
        return "The Dormant Minimalist", "Keeps the card, barely uses it."
    if freq > 0.4:
        return "The Active Everyday Spender", "Engaged, reliable card user."
    return "The Occasional User", "Light, sporadic engagement."


def _cc_traits(z):
    def g(n): return z.get(n, 0.0)
    t = []
    if g("PURCHASES") > 0.8: t.append("very high overall spend")
    elif g("PURCHASES") < -0.4: t.append("low purchase activity")
    if g("CASH_ADVANCE") > 0.8: t.append("heavy reliance on cash advances")
    if g("PURCHASES_FREQUENCY") > 0.6: t.append("frequent, habitual card use")
    elif g("PURCHASES_FREQUENCY") < -0.4: t.append("infrequent card use")
    if g("INSTALLMENTS_PURCHASES") > 0.6: t.append("a lean toward installment purchases")
    if g("ONEOFF_PURCHASES") > 0.8: t.append("large one-off purchases")
    if g("BALANCE") > 0.6: t.append("a high revolving balance")
    if g("PRC_FULL_PAYMENT") > 0.5: t.append("a habit of paying the balance in full")
    elif g("PRC_FULL_PAYMENT") < -0.3: t.append("rarely paying the balance in full")
    if g("CREDIT_LIMIT") > 0.6: t.append("high credit limits")
    return t


_CC_PLAYS = {
    "The Cash-Advance Revolver": "Manage risk and offer a healthier path: balance-transfer or structured installment offers to convert costly cash-advance debt, with credit-line monitoring.",
    "The Premium Power-Spender": "Retain and reward: premium rewards tiers, concierge perks, and early-access offers to protect share-of-wallet — losing them is expensive.",
    "The Everyday Installment Shopper": "Grow share-of-wallet: category cashback and buy-now-pay-later promotions that match existing behavior, plus limit increases.",
    "The Dormant Minimalist": "Activate, don't acquire: bounded first-purchase incentives and recurring-bill nudges to convert idle accounts into everyday spenders.",
    "The Active Everyday Spender": "Deepen engagement with targeted rewards and gentle upsell toward premium tiers.",
    "The Occasional User": "Re-engage with reminders, targeted offers, and a reason to make the card top-of-wallet.",
}


def _cc_persona(seg, seg_means, z, size, total):
    name, tagline = _cc_name(z)
    traits = _cc_traits(z)
    trait_txt = ", ".join(traits) if traits else "a balanced, average behavioral profile"
    pct = 100 * size / total

    def s(n, money=True):
        if n in seg_means.index:
            return _fmt_money(seg_means[n]) if money else f"{seg_means[n]:.2f}"
        return "n/a"

    desc = (
        f"{name} makes up {pct:.1f}% of the book ({size:,} customers). "
        f"Behaviorally, this group is defined by {trait_txt}. "
        f"Over the period they purchased about {s('PURCHASES')} and took "
        f"{s('CASH_ADVANCE')} in cash advances, carrying a typical balance of "
        f"{s('BALANCE')}. Every figure here is the segment's own average."
    )
    key_stats = []
    for f, money in [("PURCHASES", True), ("CASH_ADVANCE", True), ("BALANCE", True),
                     ("PURCHASES_FREQUENCY", False), ("CREDIT_LIMIT", True)]:
        if f in seg_means.index:
            key_stats.append([_title(f), s(f, money)])
    return {
        "segment": int(seg), "name": name, "tagline": tagline,
        "size": int(size), "pct": round(pct, 1),
        "color": PALETTE[int(seg) % len(PALETTE)],
        "description": desc, "marketing_play": _CC_PLAYS.get(name, ""),
        "key_stats": key_stats,
    }


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------
def generate_persona(seg, seg_means, overall_means, overall_std, size, total,
                     credit_card=None):
    z = _zprofile(seg_means, overall_means, overall_std)
    if os.environ.get("PERSONAE_USE_LLM") == "1":
        try:
            return _llm_persona(seg, seg_means, z, size, total)
        except Exception:
            pass
    if credit_card is None:
        credit_card = is_credit_card_schema(seg_means.index.tolist())
    if credit_card:
        return _cc_persona(seg, seg_means, z, size, total)
    return _generic_persona(seg, seg_means, z, size, total)


def _llm_persona(seg, seg_means, z, size, total):
    import anthropic, json
    profile = {k: round(float(v), 2) for k, v in seg_means.items()}
    prompt = (
        "You are a marketing analyst. Given this customer segment's AVERAGE "
        "statistics, write a persona. Rules: every claim must be grounded in "
        "the numbers below; do not invent demographics. Return JSON with keys "
        "name, tagline, description, marketing_play.\n\n"
        f"Segment size: {size} of {total} customers.\n"
        f"Average statistics: {json.dumps(profile)}\n"
        f"Standardized deviations: {json.dumps({k: round(float(v),2) for k,v in z.items()})}"
    )
    client = anthropic.Anthropic()
    msg = client.messages.create(model="claude-3-5-sonnet-latest", max_tokens=600,
                                 messages=[{"role": "user", "content": prompt}])
    data = json.loads(msg.content[0].text)
    data.update({"segment": int(seg), "size": int(size),
                 "pct": round(100 * size / total, 1),
                 "color": PALETTE[int(seg) % len(PALETTE)]})
    return data


def generate_all(result):
    means, sizes, overall = result["means"], result["sizes"], result["overall"]
    overall_std = result.get("overall_std")
    if overall_std is None:
        overall_std = means.std(ddof=0).replace(0, 1e-9)
    total = int(sizes.sum())
    cc = is_credit_card_schema(list(means.columns))
    return [generate_persona(seg, means.loc[seg], overall, overall_std,
                             int(sizes[seg]), total, credit_card=cc)
            for seg in means.index]
