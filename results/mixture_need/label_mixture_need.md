# Mixture Need Analysis (label target)

Held-out validation values were fit offline with:
- one Logistic distribution
- one 3-component Logistic mixture

## helpfulness
- Samples: 1038
- Single Logistic avg NLL: 1.6423
- 3-component mixture avg NLL: -4.6923
- Delta avg NLL (single - mix3): 6.3347
- BIC: single=3423.39, mix3=-9685.68
- Conclusion: 3-component mixture clearly preferred

## correctness
- Samples: 1038
- Single Logistic avg NLL: 1.6197
- 3-component mixture avg NLL: -3.2198
- Delta avg NLL (single - mix3): 4.8395
- BIC: single=3376.38, mix3=-6628.74
- Conclusion: 3-component mixture clearly preferred

## coherence
- Samples: 1038
- Single Logistic avg NLL: 0.9088
- 3-component mixture avg NLL: 6.6941
- Delta avg NLL (single - mix3): -5.7853
- BIC: single=1900.55, mix3=13952.44
- Conclusion: single Logistic sufficient

## complexity
- Samples: 1038
- Single Logistic avg NLL: 1.0910
- 3-component mixture avg NLL: 4.8292
- Delta avg NLL (single - mix3): -3.7382
- BIC: single=2278.80, mix3=10080.89
- Conclusion: single Logistic sufficient

## verbosity
- Samples: 1038
- Single Logistic avg NLL: 1.1409
- 3-component mixture avg NLL: -4.0754
- Delta avg NLL (single - mix3): 5.2163
- BIC: single=2382.46, mix3=-8404.94
- Conclusion: 3-component mixture clearly preferred

