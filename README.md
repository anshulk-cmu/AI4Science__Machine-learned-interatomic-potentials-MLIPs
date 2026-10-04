# Local Composition Routing for Mixture-of-Experts Interatomic Potentials

Local-composition routing for mixture-of-experts interatomic potentials, benchmarked against UMA on OMol25.

Course project for CMU 11-785 Introduction to Deep Learning, Fall 2026, AI4Science track (machine-learned interatomic potentials).

## Idea

UMA routes its Mixture of Linear Experts (MoLE) with an average element embedding taken over the whole structure, with no distance cutoff. For two fragments of different composition, this average differs from that of either fragment, so the energy is not additive when the fragments are far apart (size consistency). We propose restricting the routing average to a smooth, bounded neighborhood around each atom, and test whether this keeps the accuracy that routing adds.

## Plan

- Backbone: UMA's eSEN-based architecture, via the `fairchem` implementation. Only the router changes between configurations.
- Data: OMol25, official splits.
- Configurations: dense baseline, UMA-style global routing, element-only routing, local routing at two radii, and an isolation control.
- Measures: energy and force error, the additivity residual on neutral singlet fragment pairs, spurious host-atom forces, and dissociation curves.

## Status

Early stage. The proposal is submitted. No training code yet.

## Team

| Name | Email | GitHub |
|---|---|---|
| Anshul Kumar | anshulk@andrew.cmu.edu | [anshulk-cmu](https://github.com/anshulk-cmu) |
| Pragati Bhattad | pbhattad@andrew.cmu.edu | [aptenosys](https://github.com/aptenosys) |
| Yiwei Xia | yiweixia@andrew.cmu.edu | [sdfIvy](https://github.com/sdfIvy) |
