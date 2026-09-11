# E-Commerce Data Pipeline

## Elite Tech Internship – Data Science Task 1

### Project Overview

This project implements an ETL (Extract, Transform, Load) data pipeline for e-commerce data using Python and Pandas.

The pipeline loads raw e-commerce datasets, cleans and transforms the data, combines related datasets, performs feature engineering and scaling, validates the processed data, and exports clean datasets for further analysis.

## Dataset

The project uses a public e-commerce dataset containing:

- List of Orders
- Order Details
- Sales Target

### Dataset Statistics

| Dataset | Records |
|---|---:|
| List of Orders | 560 |
| Order Details | 1500 |
| Sales Target | 36 |

## Pipeline Workflow

```text
Raw Data
   ↓
Data Loading
   ↓
Data Cleaning
   ↓
Duplicate Removal
   ↓
Missing Value Handling
   ↓
Data Type Conversion
   ↓
Data Transformation
   ↓
Dataset Merging
   ↓
Feature Engineering
   ↓
Data Validation
   ↓
Feature Scaling
   ↓
Export Clean Data