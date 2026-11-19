"""build_cap_sequence: the array caps a sweep maps under, all powers of two."""


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def build_cap_sequence(min_cap: int, max_cap: int) -> list:
    """Every power of two from the smallest one at or above min_cap (the fewest arrays
    the model fits in) to the smallest one at or above max_cap (enough for every op at
    its largest pf)."""
    if min_cap <= 0:
        raise ValueError(f"min_cap must be positive, got {min_cap}")
    if max_cap < min_cap:
        raise ValueError(f"max_cap={max_cap} is below min_cap={min_cap}")
    caps = [_next_power_of_two(min_cap)]
    while caps[-1] < max_cap:
        caps.append(caps[-1] * 2)
    return caps


def select_five_caps(caps: list) -> list:
    """Five caps of a build_cap_sequence: its smallest and largest, and three between at a
    quarter, half and three quarters of the way in exponent, each rounded to the nearest
    power of two (halves up), so the steps are as equal as powers of two allow. Returns
    caps unchanged if there are 5 or fewer."""
    if len(caps) <= 5:
        return list(caps)
    span = len(caps) - 1
    # floor(k * span / 4 + 1/2), in integers
    return [caps[(2 * k * span + 4) // 8] for k in range(5)]
