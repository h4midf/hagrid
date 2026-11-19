"""Parameters of the macro_logic circuits, from one 1F1C config's design and SPICE
results (targets/1fefet1c/<config>, config = F1C_CONFIG, see f1c_io.py): shared by the
characterize and test scripts. The config's synthesized results and its accumulator
SRAM bank config live in <config>/macro_logic/ (OUT_DIR)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from f1c_io import CONFIG, CONFIG_DIR, DESIGN, load_result  # noqa: E402

OUT_DIR = str(CONFIG_DIR / "macro_logic")

_mac = load_result("mac")
if not _mac:
    raise FileNotFoundError(f"{CONFIG}: results/mac.json is missing (spice/mac/*.py)")

ROWS = DESIGN["array"]["rows"]
COLS = DESIGN["array"]["cols"]
CODE = DESIGN["adc"]["bits"]
CONV_CYCLES = DESIGN["adc"]["cycles_per_conversion"]
BITS_PER_CELL = DESIGN["cell"]["bits_per_cell"]

# ISAAC's weight-flip encoding (design.json "encoding"; the default design point, Laleni's
# macro, has none): a flipped cell's code is turned back into (2^w - 1) * U - code, U the
# unit column's code (its cells at level 1, so U = the number of rows whose input bit is 1)
ENCODED = "encoding" in DESIGN
if ENCODED:
    _enc = DESIGN["encoding"]
    if (_enc["scheme"], _enc["unit_columns"], _enc["unit_cell_level"], _enc["flip_bits_per_column"]) != \
            ("ISAAC weight flip", 1, 1, 1):
        raise ValueError(f"{CONFIG}: the shift-and-add implements one unit column at level 1 and one flip bit "
                         f"per column, design.json has {_enc}")
    if ROWS > (1 << CODE) - 1:
        raise ValueError(f"{CONFIG}: the unit column's sum (up to {ROWS}) does not fit a {CODE}-bit code")
    CELL_MAX = ((1 << BITS_PER_CELL) - 1) * ROWS    # a flipped cell's corrected value, at most
else:
    CELL_MAX = (1 << CODE) - 1
CV = CELL_MAX.bit_length()                          # bits of one cell's (corrected) value

# shift-and-add: an int8 weight over cells of BITS_PER_CELL bits; inputs of INPUT_BITS bits
WEIGHT_BITS = 8
INPUT_BITS = 8
CELLS = WEIGHT_BITS // BITS_PER_CELL
OUTS = COLS // CELLS
MAX_PARTIAL = sum(CELL_MAX << (BITS_PER_CELL * k) for k in range(CELLS))
PART = MAX_PARTIAL.bit_length()
ACC = PART + INPUT_BITS
# outputs per conversion window: the ADC codes stay in the output registers for one
# conversion, so every output must finish inside it: the fewest datapaths that divide
# the outputs evenly (one output per datapath per slot) in at most CONV_CYCLES slots
UNITS = next(u for u in range(-(-OUTS // CONV_CYCLES), OUTS + 1) if OUTS % u == 0)

# WMU readout: a 32-bit value over cells of BITS_PER_CELL bits; the read differences
# (code2 - code1) of each stored level over the capacitor read simulation's cases (one
# per comparator offset; a level may land on either side of a code boundary). Neighboring
# levels must not overlap; each threshold is halfway between a level's largest
# difference and the next level's smallest, rounded up
VALUE_BITS = 32
CELLS_PER_VALUE = VALUE_BITS // BITS_PER_CELL
_diffs = {}
for case in _mac["cap_read"]["by_case"]:
    _diffs.setdefault(case["level"], set()).add(int(case["read_value_codes"]))
LEVEL_DIFFS = [sorted(_diffs[level]) for level in sorted(_diffs)]
for _lo, _hi in zip(LEVEL_DIFFS, LEVEL_DIFFS[1:]):
    if max(_lo) >= min(_hi):
        raise ValueError(f"{CONFIG}: cap_read levels overlap, read differences {LEVEL_DIFFS}")
THRESHOLDS = [-(-(max(a) + min(b)) // 2) for a, b in zip(LEVEL_DIFFS, LEVEL_DIFFS[1:])]
