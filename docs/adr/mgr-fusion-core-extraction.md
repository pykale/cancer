# Migration record: Fusion extraction compatibility boundary

- **Date:** 2026-10-21
- **Status:** Baseline accepted; core-backed adoption proposed.
- **Tracking:** [KaleCancer #28](https://github.com/pykale/cancer/issues/28). The issue owns the adoption checklist, release dependency and validation evidence.

## Decision

Move generic fusion operations into PyKale while preserving KaleCancer's public classes, constructor signatures, configuration paths and custom extension interfaces. PyKale supplies generic `FusionMethod`, tensor-based `FusionContext`, `NamedInputConcat`, `MaskedMean` and `MajorityVote`; the distinct concatenation name preserves PyKale's existing `Concat` API.

Keep one inheritance path: KaleCancer's `FusionMethod` extends the core base, and its concrete methods borrow the core's stateless `defined` and `forward` implementations. Existing extension helpers wrap the public `stack_present` and `masked_mean` functions. This avoids multiple-inheritance method conflicts and duplicate tensor algorithms.

Retain KaleCancer's pandas-based `FusionContext` and application checks: fusion stages, required patients, head compatibility, prediction columns and the Cox branch-completeness safeguard. The pandas context is not passed to core's `check`. Set `validate_inputs = False` on the KaleCancer base because model checks and the dry run retain validation responsibility. Early concatenation continues to pass raw NaN cells to encoders that handle them.

Preserve float32 and float64 baseline behaviour. Half-precision means accumulate in float32 before returning the input dtype; half-precision voting returns float32 scores to retain the probability tie-breaker. Treat these as explicit numerical changes, rather than claiming complete dtype parity.

The adapter PR targets `migration/pykale-core`. Adoption requires a released PyKale API and matching dependency metadata before merging. KaleCancer main is updated only when the complete migration is ready.

## Update: endpoint architecture

- **Date:** 2026-11-24
- **Context:** [ADR 0005](0005-endpoint-architecture.md) (#23) splits fusion into `TensorFusion`, which combines tensors in early and intermediate fusion, and `OutputFusion`, which combines the branches' typed outputs in late fusion. It replaces the pandas `FusionContext` and `check` with hooks that run when their information exists: `output_type`, `check_branches` and `check_widths`.

KaleCancer's `FusionMethod` no longer extends the core base. Core's `FusionMethod.forward` takes and returns tensors, which an `OutputFusion` does not, so extending it would give late fusion methods a contract they break. The delegation of the tensor algorithms stays: `Concat` and `MaskedMean` borrow core's `defined`, and `Concat` its `forward`. `MaskedMean` runs core's `forward` on tensors, `MajorityVote` runs core's `forward` on its branches' probabilities, and the helpers still wrap `stack_present` and `masked_mean`. `validate_inputs = False` stays on the KaleCancer base, because the borrowed operations read it. The numerical behaviour above is unchanged.
