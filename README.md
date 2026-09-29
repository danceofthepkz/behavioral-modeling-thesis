# Behavioral Dynamics in Auditory Categorization

Comparing continuous adaptation and latent strategy switching in trial-by-trial decision behavior.

## Research Question

Animals gradually change their choices during learning, but observed behavior alone may not reveal whether they are continuously adapting or switching between discrete strategies.

This project asks:

> When several models predict the same choices, what evidence distinguishes the decision processes they imply?

## Motivation

A model can predict choices well while supporting the wrong explanation of the underlying behavior. I compare models with different assumptions about how decision strategies change over time.

## My Role

I maintain the thesis analysis workflow in a collaborative lab repository. My work includes:

- implementing and evaluating trial-level behavioral models;
- comparing fixed, continuously varying, and latent-state explanations;
- using held-out prediction and model comparison to evaluate competing accounts;
- analyzing choice-history effects and differences across individual animals;
- documenting weekly analyses and preparing interpretable figures.

## Model Comparison Framework

| Model | Assumption | Question it tests |
| --- | --- | --- |
| Static logistic regression | Behavioral weights remain fixed | Is one stable strategy sufficient? |
| Dynamic logistic regression (PsyTrack) | Weights change smoothly over time | Does behavior reflect continuous adaptation? |
| GLM-HMM | Behavior switches among discrete latent states | Do apparent changes reflect strategy switching? |
| Reinforcement learning | Choices update through trial-by-trial feedback | Can learning dynamics explain the observed choices? |

## Current Status

This is an ongoing undergraduate thesis project. The current analysis compares model fit, held-out prediction, history effects, and inferred behavioral dynamics across individual animals.

## Public Release Plan

This repository will contain only materials approved for public release:

- `figures/` - approved summary figures;
- `src/` - cleaned analysis code;
- `notebooks/` - reproducible demonstrations using non-sensitive or synthetic data;
- `docs/` - methods notes and model descriptions.

The interactive article uses the previously published old-mouse dataset, including the GS027 stimulus-category and choice sequences approved for this public edition. Newer unpublished experimental data and internal lab materials are not included.

## Acknowledgments

This work is conducted with Prof. Joshua Goldwyn at Swarthmore College. The active research workflow is maintained in a private collaborative repository.

## Project Status

Work in progress. Results and code will be added after review and approval.

## Interactive article

The first public edition is available at:

https://danceofthepkz.github.io/behavioral-modeling-thesis/auditory_categorization_learning/

The English article contains five connected sections: recorded learning in 19 mice, a reinforcement learning simulation with synchronized learning curves, conditional prediction versus rollout diagnostics, static/dynamic/latent-state explanations, and choice-history analyses. It follows a continuous article layout inspired by Distill.

### Run locally

```sh
python3 -m http.server 8000 --directory docs
```

Open `http://localhost:8000/auditory_categorization_learning/`. The website is static; no build step is needed. D3 and Three.js are loaded from version-pinned CDNs.

### Publish

GitHub Pages is configured to deploy the `docs` directory on `main`. The repository name stays unchanged; `auditory_categorization_learning` is the article path. Publishing does not require a backend.

### Data and interpretation

The website's `results.json` includes behavioral aggregates, published GS027 category/choice sequences, existing model comparison results, and source-file SHA-256 checksums. The 20-run rollout diagnostic was computed from existing GS027 parameters without refitting. The visual mouse and apparatus are illustrative, not a biomechanical or connectome model. Random-trial cross-validation is not a forecast of future training. Inferred GLM-HMM states are not directly observed psychological states.
