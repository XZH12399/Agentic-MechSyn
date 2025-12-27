# MechGraph Generator

A physics-informed data synthesis framework for generating large-scale mechanism datasets from a few seed examples.

## Features
- **Physics-based Repair**: Uses gradient-based optimization to repair kinematic constraints after geometric mutation.
- **Type-Preserving**: Maintains specific geometric conditions for overconstrained mechanisms (e.g., Bennett, Sarrus).
- **Auto-Labeling**: Automatically generates task descriptions (prompts) based on the kinematic analysis of generated mechanisms.

## Usage
1. Place your seed mechanism JSONs in `seeds/`.
2. Run the generator:
   ```bash
   python build_dataset.py
   ```
3. The dataset will be saved to output/.