"""Bounded comparison-shape repair guidance without echoing model content."""


COMPARISON_FIELDS = (
    "original_step", "repeated_step", "original_episode", "repeated_episode", "result",
)
COMPARISON_SHAPE_PREFIX = (
    "Comparison needs exactly original/repeated steps, original/repeated episodes, and result."
)
MAX_DIAGNOSTIC_CHARS = 640
_ASSESSMENT_CONDITION = "After the repaired response meets its assessment requirements, "
_JSON_TYPES = {
    type(None): "null", bool: "boolean", int: "integer", float: "number",
    str: "string", list: "array", dict: "object",
}


def comparison_shape_diagnostic(value, *, eligible_pair_count: int) -> str:
    """Describe only shape metadata and the independently known evidence phase.

    Unknown key names and all values are excluded. Count magnitude is omitted;
    only its zero/positive phase selects guidance. No action, status, or learned
    claim is recommended, and no eligible evidence is invented.
    """
    if type(eligible_pair_count) is not int or eligible_pair_count < 0:
        raise ValueError("eligible_pair_count must be a nonnegative integer, excluding bool.")
    kind = _JSON_TYPES.get(type(value), "non-JSON")
    # JSON decoding produces built-in dictionaries. Unsupported subclasses are
    # classified without invoking their custom keys/len/string representations.
    if type(value) is dict:
        known_keys = {key for key in value if type(key) is str and key in COMPARISON_FIELDS}
        known = [field for field in COMPARISON_FIELDS if field in known_keys]
        unknown_count = len(value) - len(known)
    else:
        known, unknown_count = [], 0
    missing = [field for field in COMPARISON_FIELDS if field not in known]
    guidance = (
        "no eligible completed assessed cross-episode pair exists in this note's cited evidence; comparison must be JSON null for this update."
        if eligible_pair_count == 0 else
        "a recorded pair is eligible in the cited evidence; comparison may be JSON null or copy one eligible pair's exact five fields."
    )
    # Every interpolated name comes from the fixed canonical field whitelist;
    # unknown_count is bounded by a real built-in dictionary's cardinality.
    return (
        f"{_ASSESSMENT_CONDITION}{guidance} {COMPARISON_SHAPE_PREFIX} JSON type: {kind}. "
        f"Known fields: {', '.join(known) or 'none'}. "
        f"Missing fields: {', '.join(missing) or 'none'}. "
        f"Unknown field count: {unknown_count}."
    )
