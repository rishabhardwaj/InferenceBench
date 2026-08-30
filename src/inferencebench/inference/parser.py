from __future__ import annotations

from inferencebench.domain import CustomerLabel, ParseStatus
from inferencebench.inference.domain import OutputNormalization, OutputParseResult


_LABELS = {label.value: label for label in CustomerLabel}
_ASCII_LOWER_TRANSLATION = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"
)
_WRAPPERS = {
    "'": OutputNormalization.SINGLE_QUOTE_WRAPPER,
    '"': OutputNormalization.DOUBLE_QUOTE_WRAPPER,
    "`": OutputNormalization.BACKTICK_WRAPPER,
}


def parse_customer_label(raw_output: str) -> OutputParseResult:
    if raw_output in _LABELS:
        return OutputParseResult(
            parser_version="bare-label-parser-v1",
            parse_status=ParseStatus.EXACT,
            parsed_label=_LABELS[raw_output],
            normalizations=(),
        )

    value = raw_output
    normalizations: list[OutputNormalization] = []

    stripped = value.strip()
    if stripped != value:
        normalizations.append(OutputNormalization.SURROUNDING_WHITESPACE)
        value = stripped

    ascii_lowered = value.translate(_ASCII_LOWER_TRANSLATION)
    if ascii_lowered != value:
        normalizations.append(OutputNormalization.ASCII_CASE)
        value = ascii_lowered

    value, wrapper_normalization = _remove_matching_wrapper(value)
    if wrapper_normalization is not None:
        normalizations.append(wrapper_normalization)

    if value.endswith("."):
        value = value[:-1]
        normalizations.append(OutputNormalization.TERMINAL_PERIOD)

    # Accept a terminal period outside a matching wrapper, for example `"bug".`.
    # Only one wrapper and one period are ever removed.
    if wrapper_normalization is None:
        value, wrapper_normalization = _remove_matching_wrapper(value)
        if wrapper_normalization is not None:
            normalizations.append(wrapper_normalization)

    parsed_label = _LABELS.get(value)
    if parsed_label is None:
        return OutputParseResult(
            parser_version="bare-label-parser-v1",
            parse_status=ParseStatus.INVALID,
            parsed_label=None,
            normalizations=tuple(normalizations),
        )
    return OutputParseResult(
        parser_version="bare-label-parser-v1",
        parse_status=ParseStatus.NORMALIZED,
        parsed_label=parsed_label,
        normalizations=tuple(normalizations),
    )


def invalid_parse_result() -> OutputParseResult:
    return OutputParseResult(
        parser_version="bare-label-parser-v1",
        parse_status=ParseStatus.INVALID,
        parsed_label=None,
        normalizations=(),
    )


def _remove_matching_wrapper(
    value: str,
) -> tuple[str, OutputNormalization | None]:
    if len(value) < 2:
        return value, None
    normalization = _WRAPPERS.get(value[0])
    if normalization is None or value[-1] != value[0]:
        return value, None
    return value[1:-1], normalization
