# Byte Accuracy Guide

Byte accuracy means local parity backends produce the same canonical snapshot bytes as the official interpreter under identical seeds and actions.

## Test command

- `python -m orbit_wars_fast.test_byte_accuracy --games 20 --agents 2 --steps 200 --seed 1337`

## What is compared

Per step, the test compares canonical JSON snapshots including:

- `planets`
- `fleets`
- `initial_planets`
- `next_fleet_id`
- `comets`
- `comet_planet_ids`
- `angular_velocity`
- `status`
- `reward`

## Determinism setup

The test ensures all compared backends receive:

- Same game seed
- Same per-step action list

By default this includes:

- `ReferenceOrbitWarsSimulator`
- `FastOrbitWarsSimulator`
- `JAXParityOrbitWarsSimulator`

You can skip jax parity validation with:

- `--skip-jax-parity`

## Mismatch handling

On mismatch, the test writes a debug file:

- `orbit_wars_fast/mismatch_game_<g>_step_<s>.json`

This includes:

- seed, step, actions
- reference snapshot
- fast snapshot

## Practical advice

- Keep Python and dependency versions stable when regression testing.
- Run both 2-player and 4-player suites.
- Use larger step counts for stronger confidence.
