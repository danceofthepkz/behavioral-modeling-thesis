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

Raw experimental data and internal lab materials are not included.

## Acknowledgments

This work is conducted with Prof. Joshua Goldwyn at Swarthmore College. The active research workflow is maintained in a private collaborative repository.

## Project Status

Work in progress. Results and code will be added after review and approval.
