# Sentinel

## Main Idea
Confidence-scored, root-cause-aware anomaly detection for autonomous lunar descent.

## Problem
Onboard fault detection during lunar descent gives a binary anomaly signal with 
no confidence score and no explanation — this has contributed to real 
mission failures (Luna-25, Chandrayaan-2, HAKUTO-R M1, Resilience).

## Solution
A monitoring layer that outputs three things per detected anomaly: a calibrated 
confidence score, ranked root-cause attribution across sensor channels, and a 
plain-language recommendation.

## Architecture
- `simulator/` — physics-based descent simulation + fault injection
- `backend/` — detection, confidence calibration, and attribution pipeline
- `dashboard/` — live replay visualization

## Built with
IBM Bob (scaffolding, pipeline code, watsonx.ai integration) + watsonx.ai 
(Granite Time Series for forecasting, Granite instruct for explanation generation)