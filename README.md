# Real-Time ECG Signal Analysis and Classification

Advanced Python implementation for ECG signal processing, feature extraction, and machine learning-based classification with progressive noise simulation and real-time visualization.

![ECG Analysis Demo](ecg_analysis_animation.gif)

## Table of Contents

1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Features](#features)
6. [Algorithm](#algorithm)
7. [Results](#results)
8. [Important Notes](#important-notes)
9. [Contact](#contact)

## Introduction

This repository provides comprehensive tools for ECG signal analysis, designed to evaluate algorithm robustness under varying noise conditions. The system generates realistic ECG signals, applies progressive noise testing, and provides real-time feedback through animated visualizations.

## Dependencies

To run this project, you'll need to have the following dependencies installed:

- **NumPy** (Version 1.21.3)
- **Pandas** (Version 1.3.4)
- **Matplotlib** (Version 3.4.3)
- **SciPy** (Version 1.7.3)
- **Scikit-learn** (Version 0.24.2)
- **TensorFlow** (Version 2.6.0)
- **Neurokit2** (Version 0.1.1)

You can install these dependencies using pip:

pip install -r requirements.txt

## Installation

1. Clone or download this repository to your local machine
2. Install the required dependencies (see Dependencies section above)
3. Ensure you have the pre-trained TensorFlow model in the ./model/ directory

## Usage

To use this code, follow these steps:

1. Ensure you have the required dependencies installed
2. Clone or download this repository to your local machine
3. Run the Python script:

python animation_real_time_with_SNR.py

This script will generate an animated plot showing the ECG signal analysis in real-time.

**Output Files:**
- ecg_analysis_animation.mp4 - High-quality MP4 animation (if FFmpeg available)
- ecg_analysis_animation.gif - Fallback GIF animation (if Pillow available)
- Console output with detailed analysis statistics

## Features

### Core Capabilities
- **Realistic ECG Simulation**: Generates physiologically accurate ECG signals with natural heart rate variations
- **Progressive Noise Testing**: Implements comprehensive noise schedule to test algorithm robustness (0.01 to 2.00 STD)
- **Real-time Processing**: Extracts features from 30-second sliding windows using cumulative signal processing
- **Machine Learning Classification**: Uses pre-trained TensorFlow model with rolling normalization
- **8-Panel Real-time Visualization**: Creates comprehensive animated visualizations of the entire ECG analysis pipeline

### Advanced Signal Processing
- Bandpass filtering (0.25-25 Hz) with ECG cleaning
- R-peak detection using NeuroKit2 advanced algorithms
- Multi-component noise simulation including Gaussian noise, high-frequency muscle artifacts, and low-frequency baseline wander
- Heart rate variability (RMSSD) calculation with outlier removal
- Real-time signal-to-noise ratio (SNR) computation

### Visualization Components
The animation provides 8 synchronized subplots:
1. Current ECG window display (10-second tail)
2. Cumulative ECG signal timeline with highlighted current segment
3. Real-time Signal-to-Noise Ratio tracking (0-30 dB range)
4. Mean heart rate trends with 70 BPM reference line
5. Maximum heart rate values over time
6. Minimum heart rate values over time
7. Heart rate variability (RMSSD) showing cardiac autonomic function
8. ML prediction probabilities with 0.5 threshold indicator

## Algorithm

### Signal Generation Process
The system generates realistic ECG signals incorporating:
- Base heart rate of 70 BPM with natural individual variations
- Respiratory sinus arrhythmia (breathing-induced heart rate changes)
- Long-term trends simulating activity or stress effects
- Random walk components for physiological drift

### Progressive Noise Schedule
Implements sophisticated noise testing:
- Frames 0-20: Clean signal (0.01 STD)
- Frames 20-50: Low noise (0.05 STD)
- Frames 50-100: Moderate to high noise (0.15-0.30 STD)
- Frames 100-200: Extreme noise testing (0.45-1.00 STD)
- Extended frames: Stress testing up to 2.00 STD with recovery phases

### Feature Extraction Pipeline
- 30-second sliding window feature extraction
- Cumulative signal processing that builds continuously like real-time systems
- Feature set includes HR mean/max/min, HRV (RMSSD), and SNR
- Quality control with z-score outlier removal and validation

### Machine Learning Classification
- Rolling normalization using cumulative data history for standardization
- Pre-trained TensorFlow saved model for ECG classification
- Real-time prediction with continuous probability assessment
- Adaptive scaling with StandardScaler applied to growing datasets

## Results

### Performance Expectations
The system provides comprehensive analysis with expected performance:
- **Clean Signal Performance**: >95% feature extraction success rate, SNR >20 dB
- **Moderate Noise Conditions**: 80-90% success rate, SNR 10-20 dB
- **High Noise Conditions**: 60-80% success rate, SNR 5-15 dB
- **Extreme Noise Conditions**: <60% success rate, SNR <10 dB

### Animation Output
The generated animation (ecg_analysis_animation.gif or .mp4) demonstrates:
- Real-time ECG signal processing under progressive noise conditions
- Feature extraction success rates across different noise levels
- Machine learning prediction confidence evolution
- Visual representation of algorithm robustness testing

### Analysis Features
The real-time visualization provides:
- Progressive noise level indicators
- Cumulative statistics display
- Visual parameter tracking
- ML confidence assessment with threshold indicators
- Interactive timeline showing ECG analysis evolution

The animated plot provides real-time insights into ECG signal analysis. You can observe changes in heart rate statistics, SNR degradation patterns, heart rate variability trends, and the model's classification probability as the simulation progresses through different noise conditions.

## Important Notes

### Pre-trained Model Requirements
**CRITICAL**: This implementation requires a pre-trained TensorFlow model that is **NOT included** in this repository. The model is loaded from `./model/` directory and is essential for the machine learning classification functionality.

**Model Details:**
- The model is trained specifically for ECG signal classification
- All predictions and classification results shown in the animation are based on this pre-trained model
- The model expects 4 input features: HR mean, HR max, HR min, and HRV (RMSSD)
- Model outputs probability values between 0 and 1 for binary classification

**Model Access:**
- The pre-trained model is **proprietary** and not publicly available
- To obtain access to the model file, you **MUST contact the author** directly
- Without the model, the script will fail to run and generate errors
- The model cannot be redistributed without explicit permission

**Contacting for Model Access:**
Please reach out to **murat.kucukosmanoglu@dprime.ai** with:
- Your intended use case for the model
- Research or educational purpose description
- Institutional affiliation (if applicable)

**Alternative Usage:**
If you want to use this code framework with your own model:
1. Train your own TensorFlow model with similar input/output structure
2. Save it in TensorFlow SavedModel format
3. Place it in the `./model/` directory
4. Ensure your model accepts 4 features and outputs single probability value

## Contact

**Project Maintainer**: Murat Kucukosmanoglu
**Email**: murat.kucukosmanoglu@dprime.ai

For any questions or inquiries, feel free to reach out for:
- **Model access requests** (Required for running the code)
- Technical assistance with implementation
- Questions about the algorithm details
- Collaboration opportunities
- Bug reports or feature requests

Please do not hesitate to contact me if you have any feedback or need assistance with using the script.

---

*This project demonstrates advanced ECG signal processing techniques suitable for research, development, and educational purposes in cardiac signal analysis.*

