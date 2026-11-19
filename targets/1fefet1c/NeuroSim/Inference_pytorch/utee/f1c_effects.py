"""The 1F1C macro's nonidealities (Laleni et al., JXCDC 2024) for NeuroSim's VGG8 inference
(modules/quantization_cpu_np_infer.py). Every effect is a switch, off by default; with all
off, every function returns exactly what NeuroSim computes without this module.

  levels      the stored value of a 2-bit cell (eq. 1): weight slice r = 0..3 is the
              Vth state for r (r = 0: the 1.2 V HVT state), the cell holds
              V_d = max(0, VDD - Vth). "main" is Sec. V's set [1.2, 0.9, 0.6, 0.3] V
              (V_d 0 / 0.1 / 0.4 / 0.7 V, nonlinear, Fig. 10(a)); "linear" is Fig. 10(b)'s
              [1.2, 0.7, 0.4, 0.1] V (0 / 0.3 / 0.6 / 0.9 V). The HVT state stores ~0, so
              the on/off floor is 0 (onoffratio). Levels are in units of NeuroSim's slice
              value: V_d(r) / V_d(3) * 3.
  vth_sigma   per-state Vth spread from Fig. 7(b) (sigma 17 / 42 / 50 mV for r = 3 / 2 / 1;
              r = 0 stores ~0), added to the stored voltage.
  fixed_adc   the column ADC with a fixed full scale instead of NeuroSim's per-tensor
              min/max range: V_BL = kappa * sum(in * V_d) / N over the N = subArray cells of
              the column (eq. 4, kappa = N C_MoM / (N C_MoM + C_par)), quantized by a
              `bits`-bit SAR with Vrefn = 0, Vrefp = 0.5 V (floor, as NeuroSim's quantizer).
  enob        the ADC's effective bits (6.4): Gaussian noise at its input of
              LSB * sqrt((2^(2 (bits - enob)) - 1) / 12), the noise that lowers a `bits`-bit
              quantizer to `enob` effective bits. With fixed_adc off, LSB is NeuroSim's.

inference.py sets the switches with configure(args) before the model runs.
"""
import math

import torch

from utee import wage_quantizer

VDD = 1.0
VTH_STATES_V = {"main": (1.2, 0.9, 0.6, 0.3), "linear": (1.2, 0.7, 0.4, 0.1)}
# Fig. 7(b): sigma of the state each slice value r is stored in (r = 0 stores ~0)
VTH_SIGMA_V = (0.0, 0.050, 0.042, 0.017)
VREFP_V = 0.5

CONFIG = {"levels": "off", "vth_sigma": False, "fixed_adc": False, "kappa": 0.69, "enob": 0.0}


def configure(args) -> None:
    CONFIG["levels"] = args.f1c_levels
    CONFIG["vth_sigma"] = bool(args.f1c_vth_sigma)
    CONFIG["fixed_adc"] = bool(args.f1c_fixed_adc)
    CONFIG["kappa"] = float(args.f1c_kappa)
    CONFIG["enob"] = float(args.f1c_enob)
    if CONFIG["levels"] not in ("off", *VTH_STATES_V):
        raise ValueError(f"f1c_levels must be off, main or linear, got {CONFIG['levels']!r}")


def _stored_v(levels: str) -> tuple:
    return tuple(max(0.0, VDD - vth) for vth in VTH_STATES_V[levels])


def _scale_set() -> str:
    """The Vth set that sets the voltage scale: the chosen one, or "main" (Sec. V's) when
    the levels are left ideal but a voltage-based effect is on."""
    return CONFIG["levels"] if CONFIG["levels"] != "off" else "main"


def onoffratio(ratio: float) -> float:
    """NeuroSim's on/off ratio; infinite (floor 0) when the 1F1C levels are on."""
    return math.inf if CONFIG["levels"] != "off" else ratio


def cell_value(remainder: torch.Tensor) -> torch.Tensor:
    """The value a cell holding weight slice `remainder` (0..3) contributes, in slice units."""
    if CONFIG["levels"] == "off" and not CONFIG["vth_sigma"]:
        return remainder
    v = _stored_v(_scale_set())
    unit = v[3] / 3.0  # volts per slice unit
    r = remainder.round().clamp(0, 3).long()
    if CONFIG["levels"] != "off":
        value = torch.tensor([x / unit for x in v], dtype=remainder.dtype, device=remainder.device)[r]
    else:
        value = remainder
    if CONFIG["vth_sigma"]:
        sigma = torch.tensor([s / unit for s in VTH_SIGMA_V], dtype=remainder.dtype, device=remainder.device)[r]
        value = value + sigma * torch.randn_like(value)
    # cells outside the mask (remainder 0 there) stay 0
    return torch.where(remainder == 0, torch.zeros_like(value), value)


def _enob_sigma_lsb(bits: int) -> float:
    if CONFIG["enob"] <= 0 or CONFIG["enob"] >= bits:
        return 0.0
    return math.sqrt((2.0 ** (2 * (bits - CONFIG["enob"])) - 1) / 12.0)


def adc(x: torch.Tensor, bits: int, rows: int) -> torch.Tensor:
    """A column partial sum `x` (slice units, summed over up to `rows` cells) after the ADC."""
    noise_lsb = _enob_sigma_lsb(bits)
    if not CONFIG["fixed_adc"]:
        if noise_lsb > 0:
            delta = (torch.max(x) - torch.min(x)).item()
            x = x + noise_lsb * (delta / 2.0 ** bits) * torch.randn_like(x)
        return wage_quantizer.LinearQuantizeOut(x, bits)
    unit = _stored_v(_scale_set())[3] / 3.0
    to_volts = CONFIG["kappa"] * unit / rows
    lsb = VREFP_V / 2.0 ** bits
    v_bl = x * to_volts
    if noise_lsb > 0:
        v_bl = v_bl + noise_lsb * lsb * torch.randn_like(v_bl)
    code = torch.clamp(torch.floor(v_bl / lsb), 0, 2.0 ** bits - 1)
    return code * lsb / to_volts
