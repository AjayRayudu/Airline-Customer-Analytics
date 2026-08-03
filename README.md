# Airline Customer Booking Prediction & Lounge Capacity Modeling

An applied data science project analyzing airline customer behaviour: predicting
which customers are likely to complete a booking, and modeling airport lounge
demand for capacity planning.

## Overview

This project has two components:

1. **Lounge Capacity Modeling**: a reusable lookup table and methodology for
   estimating lounge demand based on customer segment (cabin class, loyalty tier,
   alliance status), designed to support proactive capacity planning at busy hubs.

2. **Booking Prediction Model**: a Random Forest classifier trained on 50,000
   customer search sessions to predict booking completion, with feature engineering,
   class imbalance handling, and full evaluation (ROC-AUC, precision/recall,
   cross-validation).

## Key Results

- ROC-AUC: 0.78 | Recall (bookers): 76% | Accuracy: 69.6%
- Top predictors: customer origin country, length of stay, flight duration,
  purchase lead time

## Tech Stack

Python · pandas · scikit-learn · matplotlib · python-pptx

## Structure

1. main.py full pipeline: data prep, feature engineering, model training, evaluation
2. customer_booking.csv # raw dataset
3. Booking_Model_Summary_2pager.pptx # 2-page executive summary deck
