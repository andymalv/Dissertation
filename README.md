# What Moves Us: Studies of Propulsion Measurement in Poststroke Walking

Boston University, Sargent College of Health and Rehabilitation Sciences

## Committee
Dr. Cara L. Lewis, PhD, PT (Major Professor)

Dr. Louis N. Awad, PhD, PT, DPT

Dr. Elliot Saltzman, PhD

## Abstract
The efficiency of human walking gait is an important factor of what makes walking
such a vital part of human daily living. However, this efficiency, particularly that
of propulsion forces, is inhibited in populations with neuromotor walking deficits,
including stroke. The ability to measure propulsion forces accurately involves the
use of large and expensive equipment that is often not available for clinicians to use.
With wearable sensors and machine learning, there presents an opportunity to make
laboratory-based measurements of propulsion accessible to clinicians.

An initial step in this endeavor was to determine the most significant metrics
of propulsion as it relates to distance walked during a 6MWT, a popular outcome
measure linked to long distance walking function and increased quality of life. Thus,
the first aim of this dissertation was to evaluate the effects of propulsion metrics on
long distance walking function using statistical learning methods. The results showed
that braking magnitude and impulse of both the paretic and nonparetic limbs were
significant predictors of total distance walked, differing from the common focus on propulsion.

A following step was to confirm that these measurements can be accessible to
clinicians in a cost-effective manner. The second aim of this dissertation was to
validate the accuracy of an IMU based machine learning algorithm in estimating
propulsion metrics versus laboratory-based equipment. The results showed that both
propulsion metrics and entire APGRF curves cannot accurately be estimated with
the method used.

While the insight of braking metrics as a predictor of long distance walking function is useful, more work can be done to tailor accessible technologies to the needs of
clinicians.


## Project Organization

```
├── LICENSE            <- Open-source license
├── README.md          <- General information
│
├── Study 1            <- Multivariate multiple regression study
│   │
│   ├── T1_ExtractData.m        <- Extract data from V3D file
│   ├── T2_CalculateMetrics.m   <- Calculate APGRF metrics from treadmill data
│   ├── T3_GraphCheck.m         <- Plot APGRFs w/ metrics
│   └── TM_data_fft.m           <- Fast Fourier Transform of treadmill data 
│
├── Study 2            <- IMU based machine learning algorithm study
│   │
│   ├── figure_fix.py           <- Plot fixes for dissertation submission
│   ├── IMUFilter.m             <- Filter IMU data
│   ├── IMUStance.m             <- Trim IMU data to stance phase
│   ├── main.py                 <- Main Python file
│   └── mod.py                  <- Module containing custom functions
│
└── Writing            <- Written dissertation files

```


--------
