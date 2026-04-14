# Fairness and Reliability Evaluation in Facial Attribute Analysis Frameworks

Final year project — Trinity College Dublin

This project critically evaluates the fairness and reliability of the DeepFace facial attribute analysis framework across six intersectional demographic groups: Asian men, Asian women, Black men, Black women, White men, and White women.

Tools like DeepFace are increasingly used to extract demographic predictions from facial images, with outputs routinely treated as ground truth in downstream fairness evaluations. This project empirically demonstrates that this assumption is poorly founded.

## Four interconnected analyses were conducted:

- **Baseline evaluation** of gender, race, and age classification accuracy across demographic groups using a synthetic dataset of 240 images
- **Adversarial attack analysis** using FGSM and PGD at five perturbation magnitudes across all 240 images
- **Counterfactual explanation analysis** using semantic attribute interventions and LIME saliency mapping to explain the mechanisms behind misclassification
- **Ground truth validation** on 240 real-world images from the FairFace benchmark

## Key Findings

- DeepFace achieves 100% gender classification accuracy for all male groups but only 47.5% for Black women on synthetic images, falling to 22.5% on real-world photographs
- A pipeline discrepancy of 42.5 percentage points between full pipeline accuracy (47.5%) and direct model access (90.0%) reveals that face detection and alignment are the primary source of errors for Black women, not the gender classifier itself
- Black women are the most adversarially fragile group, requiring the smallest perturbation to flip predictions under PGD (mean minimum epsilon of 2.10)
- Gender misclassification is mechanistically associated with a collapse of upper-face attention, dropping from ~83–90% in correct predictions to ~9–20% in incorrect ones
- DeepFace's confidence on incorrect predictions for Black women (0.913) exceeds its confidence on correct predictions (0.828), rendering standard confidence-based quality filters entirely ineffective

## Dataset
Synthetic images were generated using Grok Imagine. Real-world validation used the FairFace benchmark (Karkkainen and Joo, 2021): https://github.com/joojs/fairface
