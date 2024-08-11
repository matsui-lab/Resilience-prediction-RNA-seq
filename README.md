# Resilience-prediction-RNA-seq

This repository contains scripts used in the study for predicting the resilience score of Alzheimer's disease patients using RNA-seq data and various machine learning models. The study aims to build and compare regression models to accurately predict resilience scores, using data from the MSBB and ROSMAP cohorts.

## Overview
In this research, we constructed regression models to predict resilience scores in Alzheimer’s disease patients. The resilience score is defined as the difference between the observed cognitive score and the cognitive score predicted by a regression model built using pathology scores.

## Data
Cohorts Used: MSBB (Mount Sinai Brain Bank) and ROSMAP (Religious Orders Study and Memory and Aging Project).
Resilience Score Calculation: The resilience score is derived by subtracting the cognitive score predicted by a regression model (based on pathology scores) from the observed cognitive score.

## Models
We compared the performance of the following models:

Support Vector Regression (SVR)
Linear Model
XGBoost
Random Forest
Transformer-based Model
Results
Across both MSBB and ROSMAP datasets, the SVR model demonstrated the best performance in predicting resilience scores.

## Repository Contents
This repository includes scripts for:

Data Preprocessing: Scripts to preprocess RNA-seq and related clinical data to prepare it for modeling.
Model Training and Evaluation: Scripts for training and evaluating the regression models, including SVR, linear models, XGBoost, Random Forest, and Transformer-based models.
Figures and Results: Code to generate the figures illustrating the model comparison results, as shown in the study.
Reproducibility
The provided scripts allow for the reproduction of the results as described in the paper. Ensure that you have access to the MSBB and ROSMAP datasets, and follow the instructions in the scripts to preprocess the data and run the models.
