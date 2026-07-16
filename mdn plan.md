# MDN Head Replacement for URM

## Goal

Replace URM's current Gaussian attribute head:

```python
5 attributes × (mu, sigma)
```

with:

```python
5 attributes × K mixture components
```

using:

```python
K = 3
```

and Logistic Mixture Density Networks.

---

# Output Layout

Current URM:

```python
[B, seq_len, 10]
```

meaning:

```python
[
 help_mu,
 help_sigma,

 corr_mu,
 corr_sigma,

 coh_mu,
 coh_sigma,

 comp_mu,
 comp_sigma,

 verb_mu,
 verb_sigma
]
```

---

New MDN output:

For each attribute:

```python
[
 pi1, pi2, pi3,
 mu1, mu2, mu3,
 s1,  s2,  s3
]
```

9 values per attribute.

For 5 attributes:

```python
5 * 9 = 45 outputs
```

Therefore:

```python
self.score = nn.Linear(hidden_size, 45)
```

---

# MDN Attribute Head

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ATTRIBUTES = 5
NUM_COMPONENTS = 3


class URMMDNHead(nn.Module):

    def __init__(
        self,
        hidden_size=4096,
        num_attributes=NUM_ATTRIBUTES,
        num_components=NUM_COMPONENTS,
    ):
        super().__init__()

        self.num_attributes = num_attributes
        self.num_components = num_components

        self.output_dim = (
            num_attributes *
            num_components *
            3
        )

        self.proj = nn.Linear(
            hidden_size,
            self.output_dim
        )

    def forward(self, hidden_states):

        out = self.proj(hidden_states)

        B, L, _ = out.shape

        out = out.view(
            B,
            L,
            self.num_attributes,
            3,
            self.num_components
        )

        logits_pi = out[:, :, :, 0, :]
        mu = out[:, :, :, 1, :]
        raw_s = out[:, :, :, 2, :]

        pi = F.softmax(
            logits_pi,
            dim=-1
        )

        s = F.softplus(raw_s) + 1e-6

        return pi, mu, s
```

---

# Mixture Mean

Used as attribute reward.

```python
def mixture_mean(
    pi,
    mu,
):
    """
    pi : [B,L,5,K]
    mu : [B,L,5,K]

    returns:
        [B,L,5]
    """

    return torch.sum(
        pi * mu,
        dim=-1
    )
```

---

# Mixture Variance

Used for uncertainty.

```python
def mixture_variance(
    pi,
    mu,
    s,
):

    mean = mixture_mean(
        pi,
        mu
    )

    second_moment = torch.sum(
        pi * (
            s ** 2 +
            mu ** 2
        ),
        dim=-1
    )

    return (
        second_moment -
        mean ** 2
    )
```

---

# Logistic Mixture Log Density

```python
def log_mixture_pdf(
    r,
    pi,
    mu,
    s,
):

    r = r.unsqueeze(-1)

    z = (r - mu) / s

    log_pdf = (
        -torch.log(s)
        -F.softplus(z)
        -F.softplus(-z)
    )

    log_pi = torch.log(
        pi + 1e-10
    )

    return torch.logsumexp(
        log_pi + log_pdf,
        dim=-1
    )
```

---

# MDN Loss

Used during HelpSteer2 attribute regression.

```python
def mdn_nll_loss(
    target,
    pi,
    mu,
    s,
):

    log_prob = log_mixture_pdf(
        target,
        pi,
        mu,
        s
    )

    return -log_prob.mean()
```

---

# Attribute Reward Computation

For inference:

```python
attr_scores = mixture_mean(
    pi,
    mu
)
```

Shape:

```python
[B,L,5]
```

After pooling:

```python
[B,5]
```

---

# Gating Layer

DO NOT MODIFY.

Reuse existing URM code:

```python
scores = (
    attr_scores *
    pooled_weights
).sum(dim=-1)
```

This ensures a clean comparison:

URM:
Gaussian -> Mean -> Gating

MDN-URM:
Mixture Density -> Expected Reward -> Gating

````

---

# Training Plan

Stage 1:

```python
HelpSteer2
````

Freeze:

```python
Llama backbone
```

Train:

```python
MDN head
```

Loss:

```python
mdn_nll_loss
```

---

Stage 2:

```python
Skywork Preference 80K
```

Freeze:

```python
Backbone
MDN head
```

Train:

```python
gating network
```

using original BT loss.

---

# Explicit Scope Restriction

DO NOT implement:

* ALD transport
* conformal prediction
* risk-aware score
* quantile transport
* abstention

Use only:

* mixture density likelihood
* mixture mean
* mixture variance

This is the first milestone.
