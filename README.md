# Generative Digit Modeling & Image Reconstruction

**MATLAB** · University of Patras · Machine Learning coursework

MATLAB experiments using a fixed two-layer generative neural network for handwritten-digit synthesis, restoration from corrupted observations, and recovery from low-resolution measurements.

## What this project covers

- generator with latent vector
- handwritten digit synthesis
- latent optimization
- gradient-based reconstruction
- noisy observations
- low-resolution inverse problem

## Repository layout

- `report/course-report-redacted.pdf` — student-ID-redacted copy of the submitted report with methodology, findings, plots and code appendix
- Source files:
  - `code/helpers/diff_Relu.m`
  - `code/helpers/diff_sigmoid.m`
  - `code/problem_2_1_digit_generation.m`
  - `code/problem_2_2_image_reconstruction.m`
  - `code/problem_2_3_low_resolution_reconstruction.m`

## Getting started

Place the course `.mat` inputs alongside the code or on the MATLAB path. Add `code/helpers/` to the path and run `problem_2_1_...m` through `problem_2_3_...m` as separate experiments.

### Data / software dependencies

- data21.mat
- data22.mat
- data23.mat (provided in original course but not present in the uploaded PDF)

The datasets and any large/generated artifacts have deliberately not been added to version control.

## Technical notes

Uses the generative network as a constrained image prior, optimizing a latent code rather than all output pixels.

**Source provenance and execution status:** The author confirmed that the source code in the report was tested. The separate `.m` files have been transcribed from its appendix; the original course `.mat` assets were not included in the ZIP.

## Credits

Coursework authored by Spiros Bantis, University of Patras. The report remains the reference for experimental setup and observed results. Refer to the assignment for technical details.
