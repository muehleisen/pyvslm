from enum import StrEnum


class AnalysisMode(StrEnum):
    LP           = "lp"
    LEQ          = "leq"
    OCTAVE       = "octave"
    THIRD_OCTAVE = "third_octave"
    PSD          = "psd"
    SPECTROGRAM  = "spectrogram"


class Weighting(StrEnum):
    A = "A"
    C = "C"
    Z = "Z"


class ResponseSpeed(StrEnum):
    SLOW    = "Slow"
    FAST    = "Fast"
    IMPULSE = "Impulse"


class BandResolution(StrEnum):
    OCTAVE       = "octave"
    THIRD_OCTAVE = "third"


class LeqInterval(StrEnum):
    MS_100 = "ms_100"
    SEC_1  = "sec_1"
    SEC_10 = "sec_10"
    MIN_1  = "min_1"
    MIN_15 = "min_15"
    HOUR_1 = "hour_1"


# Maps LeqInterval key → (display label, duration in seconds)
LEQ_INTERVAL_MAP: dict[LeqInterval, tuple[str, float]] = {
    LeqInterval.MS_100: ("100 ms",  0.1),
    LeqInterval.SEC_1:  ("1 sec",   1.0),
    LeqInterval.SEC_10: ("10 sec",  10.0),
    LeqInterval.MIN_1:  ("1 min",   60.0),
    LeqInterval.MIN_15: ("15 min",  900.0),
    LeqInterval.HOUR_1: ("1 hour",  3600.0),
}

# Analysis mode IDs used by the GUI button group (index → AnalysisMode)
MODE_ID_MAP: dict[int, AnalysisMode] = {
    0: AnalysisMode.LP,
    1: AnalysisMode.LEQ,
    2: AnalysisMode.OCTAVE,
    3: AnalysisMode.THIRD_OCTAVE,
    4: AnalysisMode.PSD,
    5: AnalysisMode.SPECTROGRAM,
}
