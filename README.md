# Airline-Customer-Analytics
End-to-end machine learning project predicting airline customer booking completion, plus a lounge capacity planning model built with pandas, scikit-learn, and Random Forest.

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

See `/lounge_eligibility` and `/booking_prediction` for each component, including
source code, output charts, and a one-page executive summary deck.


