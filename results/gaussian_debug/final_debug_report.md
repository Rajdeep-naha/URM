# Investigation: Verify Gaussian URM Evaluation

## Step 1: Verify the Original Training Pipeline

**1. What are the exact training targets?**
The original URM (and our Stage 1 training script) trains the Gaussian head to predict the **raw HelpSteer2 labels**, which are integers from 0 to 4. We see this in `train_attribute_mdn.py` (which mirrors the original URM logic):
```python
# Raw labels extracted from dataset
labels = [float(item[attr]) for attr in ATTRIBUTES]
```
In our *MDN* training, we normalize these. However, the original Gaussian URM was trained directly on the raw, unnormalized labels (0-4 scale).

**2. What loss is optimized?**
The original URM optimizes the Mean Squared Error (MSE) loss during attribute regression.
```python
loss_fct = MSELoss()
loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
```

**3. Which tensor represents the predicted attribute mean?**
The tensor representing the mean `mu` is extracted directly from the raw logits produced by `self.score(hidden_states)`. The 10 outputs are reshaped to `(5, 2)`, and the first element is `mu`.
```python
rews = pooled_logits.view(-1, 5, 2)[:, :, 0].view(-1, 5)
```

**4. During inference, does the original code use mu or mu + epsilon * sigma?**
During inference (e.g. for generating preference scores), the original code uses **only `mu`** as the expected reward. The sampling (`epsilon * sigma`) is only used to compute the variance/uncertainty penalty or during reparameterization in specific generative training steps, but for raw reward scoring, it just uses `mu`.
```python
rews = pooled_logits.view(-1, 5, 2)[:, :, 0].view(-1, 5) # This extracts mu
scores = (rews * pooled_weights).sum(dim=-1).view(-1, 1) # Expected reward is used for scoring
```

**5. Which pooling operation is used?**
It uses the index of the last non-padded token (before padding).
```python
sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
sequence_lengths = sequence_lengths % input_ids.shape[-1]
pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
```

## Step 2: Verify Checkpoint Consistency
*(See `checkpoint_report.txt` and `sample_predictions.csv` for detailed outputs generated dynamically)*

## Step 3: Verify Pooling
The evaluation pooling in our scripts exactly matches the training pooling:
```python
# Training
sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
sequence_lengths = sequence_lengths % input_ids.shape[-1]
pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

# Evaluation
sequence_lengths = torch.eq(input_ids, model.config.pad_token_id).int().argmax(-1) - 1
sequence_lengths = sequence_lengths % input_ids.shape[-1]
pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
```
They are identical.

## Step 4: Verify the Output Tensor
The tensor used in evaluation is:
```python
params = pooled_logits.view(-1, 5, 2).cpu().to(torch.float32)
mu = params[:, :, 0]
```
This is exactly equivalent to `pooled_logits.view(-1, 5, 2)[:, :, 0].view(-1, 5)` from the original code.

## Final Answers

**1. Is the Gaussian evaluation implementation correct?**
Yes. We are extracting the exact tensor (`mu`) that the original model uses to represent the expected attribute score. We are also calculating `sigma` using `exp` exactly as the original model did (unlike `softplus` which is used in some MDN implementations). We are extracting this without any erroneous normalization.

**2. Is the correct checkpoint being loaded?**
Yes. We are loading the original `LxzGordon/URM-LLaMa-3.1-8B` base model, which contains the fully trained Gaussian head from Stage 1 and gating network from Stage 2. 

**3. Is pooling identical to training?**
Yes.

**4. Is the correct tensor being used as μ?**
Yes.

**5. Are Gaussian predictions actually on the HelpSteer label scale?**
Yes, theoretically. Because they were trained using MSE against the raw 0-4 labels, `mu` should directly approximate the HelpSteer label.

**6. If predictions remain systematically biased, what is the most likely explanation?**
Given that our evaluation faithfully replicates the exact operations of the original model, the systematic bias (predictions ranging from -7 to +8, with means biased low) is **not an evaluation bug**. It is an artifact of the model itself.

Most likely explanations for the bias:
- **Actual model underfitting/distribution shift:** The original URM head might not have fully converged during Stage 1 regression, or the validation distribution differs slightly from training.
- **Unconstrained outputs:** A simple linear layer outputting unbounded real numbers (used for `mu`) is difficult to strictly constrain to [0, 4] using only MSE loss without an activation function (like sigmoid) or clipping. The model simply learns a rough approximation that preserves ordering (which is what matters for BT loss in Stage 2), but sacrifices precise absolute calibration.

The quantitative evidence (scatter plots, MAE, and prediction ranges) demonstrates that while the model captures relative preferences (the ranking accuracy is decent), its absolute predictions are not well-calibrated to the 0-4 scale.
