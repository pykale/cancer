# ADR 0003: Fusion extraction compatibility boundary

- **Date:** 2026-10-21
- **Status:** Baseline accepted for local development; core extraction remains proposed.
- **Scope:** Preserve fusion behaviour while migrating generic fusion classes and context to released PyKale APIs.

## Decision

Migrate generic `FusionMethod`, `FusionContext`, `Concat`, `MaskedMean` and `MajorityVote` implementations to PyKale core. This supersedes the initial primitives-only direction. Helpers support the migration but are not the complete deliverable. Use a distinct core name, provisionally `PresenceAwareConcat`, to preserve PyKale's existing `Concat` API.

Keep thin compatibility classes at the existing KaleCancer module paths, preserving public imports, constructor arguments, configuration paths, custom subclasses and model checks. Back them with core implementations through inheritance or delegation, selecting the design only after testing method resolution and `isinstance(fusion, FusionMethod)` compatibility.

Move generic presence/width context and validation to core without importing cancer targets. Replace application target references with explicit capabilities or task-neutral metadata; decide the pandas versus tensor context representation during API design. Cancer adapters retain stage policy, target checks, prediction-column conventions, required-patient semantics and the Cox branch-completeness restriction. Preserve helper signatures referenced by the extension guide through wrappers or aliases as appropriate.

## Supported contract

- `defined(present)` accepts a boolean `(patients, inputs)` matrix. Concat requires all inputs; mean and voting require at least one. `forward` receives only accepted rows.
- Named input iteration determines concatenation order and stacking order. Presence masks are matched by name. An input absent for every row can be omitted from `values` for masked mean and voting.
- Stacked helpers use `(inputs, patients, features)` values and `(inputs, patients)` masks. Masked NaNs contribute neither values nor gradients. The helper returns zero for empty rows; this does not make those rows valid predictions.
- Floating dtype, output dimensions and differentiable contributions are preserved. Voting is discrete through argmax and differentiable through its probability term away from argmax boundaries.
- Vote scores use `(votes + 0.5 * mean_probability) / (present_branches + 0.5)`. Exact score ties select the first class under argmax; the error policy rejects tied vote counts. These scores are not calibrated probabilities.

## Baseline evidence

`tests/test_fusion_parity.py` uses explicit outputs and analytic gradients for concatenation, masked mean and voting, including absent NaNs, omitted inputs, float32/float64, ties and empty helper rows. It also checks current class identity and pickle round trips. Historical pickle compatibility is not established by a same-version round trip; retain the pre-delegation revision when evaluating historical artefacts during adoption.

Existing tests remain authoritative for stage/width/head checks, Cox restrictions and custom subclasses (`test_model_components.py`), model integration (`test_model_fusion.py`), pipeline behaviour (`test_pipeline.py`) and YAML configuration (`test_config.py`). Unsupported forward inputs are not frozen into new guarantees.

## Adoption and development

This baseline changes no production code or dependencies. CI includes pushes and pull requests targeting `migration/pykale-core`, alongside main. Keep all work local until explicitly authorised to publish it.

Review the baseline independently. Next agree the core class/context contract and adapters, implement and document all five generic classes in core, then wait for a released API before cancer adopts it. Update dependency metadata and the lockfile at adoption. Keep baseline tests unchanged and run the full suite, pre-commit, YAML loading and synthetic survival checks. Add cross-version saved-model and custom-subclass verification; stable module paths alone do not establish historical pickle compatibility.

A delayed core release does not block cancer development. Rollback restores the previous local fusion implementation and dependency metadata as needed. No production class migration or dependency change is part of this baseline task.
