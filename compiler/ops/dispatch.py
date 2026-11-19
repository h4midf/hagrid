"""Config-entry lookup and impl dispatch for the compiler/ops/<kind>.py files. Each kind
registers its impls in one handler dict; dispatch() calls the one the op's config entry
names, and raises for an impl with no handler."""


def fx_name_of(op) -> str:
    """The op's fx node name (carried by copy_provenance); raises if it has none."""
    attr = op.attributes.get("fx_name")
    if attr is None:
        raise KeyError(
            f"{type(op).__name__}: missing fx_name attribute -- cannot look up its application-config entry"
        )
    return attr.data


def get_entry(config: dict, op, expected_kind: str) -> dict:
    """The config entry for the op's fx_name; raises if missing or of another kind."""
    fx_name = fx_name_of(op)
    entry = config.get(fx_name)
    if entry is None:
        raise KeyError(f"{fx_name}: no entry in application config for this {expected_kind!r} op")
    if entry.get("kind") != expected_kind:
        raise ValueError(
            f"{fx_name}: config entry kind {entry.get('kind')!r} does not match this op's own kind {expected_kind!r}"
        )
    return entry


def require(entry: dict, key: str, fx_name: str):
    if key not in entry:
        raise KeyError(f"{fx_name}: application-config entry is missing required key {key!r}")
    return entry[key]


def dispatch(op, entry: dict, rewriter, impl_handlers: dict) -> None:
    """Calls impl_handlers[entry["impl"]](op, entry, rewriter); raises for an impl with
    no handler, naming the ones there are."""
    fx_name = fx_name_of(op)
    impl = require(entry, "impl", fx_name)
    handler = impl_handlers.get(impl)
    if handler is None:
        known = sorted(impl_handlers.keys())
        raise NotImplementedError(
            f"{fx_name}: impl {impl!r} requested, but this operation only has handlers for {known}"
        )
    handler(op, entry, rewriter)
