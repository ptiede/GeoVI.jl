---
name: geovi-julia-dev
description: Use when working on the GeoVI Julia package or similar Julia implementations of geometric variational inference, especially for API design, dense white-parameter latent models, and Enzyme/Reactant/ADTypes integration.
---

# GeoVI Julia Development

Use this skill when the task is about developing or reviewing the GeoVI Julia codebase, or when working on closely related Julia implementations of geometric variational inference.

## Working Style

1. Build context from the local code, tests, and design notes before changing APIs or implementation details.
2. Think through the interface before implementing. Prefer small, explicit APIs over broad abstractions.
3. Ask focused questions when uncertainty would change the public interface or likely cause rework. Otherwise make a reasonable assumption and state it after the work.
4. Keep the code minimal and avoid unnecessary repetition. When two paths share structure, prefer one internal seam with a clear public surface.

## Modeling Assumptions

- Assume latent parameters are represented as a single dense array.
- Assume those latent parameters live in IID standard normal coordinates unless the user says otherwise.
- Prefer handling model structure through the forward map and likelihood composition, not through generic tree-shaped latent containers.

## Implementation Preferences

- When adding AD support, prefer one linearization boundary that can provide both pushforward and pullback actions.
- Preserve the explicit `fit([rng], likelihood, xi0, family, divergence, optimizer; config=...)` style unless the user asks for a different interface.
- Follow the package's existing testing style and add targeted tests for new interfaces before relying on broader end-to-end coverage.

## Validation

- Prefer running focused tests first when changing a small surface area.
- Run `julia --project=. test/runtests.jl` before wrapping up when practical.
