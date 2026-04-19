---
title: "TTT3R: recurrent state, test-time gating, and code map"
date: 2026-04-16
slug: ttt3r
description: "Notes linking CUT3R-style recurrence to TTT3R gates with pointers into Inception3D/TTT3R."
---

### Motivation
VGGT is expensive to run
CUT3R provides a persistent state but has catastrophic forgetting
**Key insight**: CUT3R ~ RNN -> We have methods to improve RNN -> Test-time Training -> Can we apply it to CUT3R?

### Rederiving with Named Tensor Notations

#### Generic formulation of point-map regressors
For each image $I_t \in R^{W \times H \times 3}$, we want to estimate poses $C_t$ , point clouds $P_t$ , and any other methods we want
VGGT (and more) can be formulated by 4 main operations

```
X_t = tokenize(I_t) # encode to tokens
S_t = update(S_t-1, X_t) # update scene tokens
Y_t = read(S_t, X_t) # aggregate scene information to image token
C_t, P_t, ..., = decode(Y_t) # decode to modality space
```
Aha, now that we have this, let's go through some common parameterization of $S_t$ and update/read operations

#### VGGT (no holds bar in memory)
```
S_t will have K_S_t and V_S_t
K_S_t # [state_len, img_len, dim]
V_S_t # [state_len, img_len, dim]


X_t # [img_len, img_dim]
Q_X_t # [img_len, dim]
K_X_t # [img_len, dim]
V_X_t # [img_len, dim]

update(S_t-1, X_t):
	K_S_t = concat(K_S_t-1, K_X_t) # [state_len', img_len, dim]
	V_S_t = concat(V_S_t-1, V_X_t) # [state_len', img_len, dim]
	# at each time, we simple increase the state by appending more KV pairs

read(S_t, X_t):
	arg_sm = Q_X_t dot(img_len, dim) K_S_t # [img_len, state_len']
	sm = softmax(state_len') (arg_sm)
	up = sm dot(state_len') V_S_t # [img_len, dim]
	Y_t = X_t + up # [img_len, dim]
```


#### CUT3R (RNN-based)

*[Figure omitted — was embedded image.]*

```
S_t # [state_len, state_dim] - only a single latent vector -> More efficient!
Q_S_t # [state_len, dim]
K_S_t # [state_len, dim]
V_S_t # [state_len, dim]


X_t # [img_len, img_dim] - image tokens, same deal
Q_X_t # [img_len, dim]
K_X_t # [img_len, dim]
V_X_t # [img_len, dim]


update(S_t-1, X_t):
	arg_sm = Q_S_t dot(dim) K_X_t # [state_len, img_len]
	sm = softmax(img_len) (arg_sm)
	up = sm dot(img_len) V_X_t # [state_len, dim]
	S_t = S_t-1 + up

read(S_t, X_t):
	arg_sm = Q_X_t dot(dim) K_S_t # [img_len, state_len]
	sm = softmax(state_len) (arg_sm)
	up = sm dot(state_len) V_S_t # [img_len, dim]
	Y_t = X_t + up
		
```


#### TTT3R (adding gradient update)

*[Figure omitted — was embedded image.]*

```
S_t # [state_len, state_dim] - only a single latent vector -> More efficient!
Q_S_t # [state_len, dim]
K_S_t # [state_len, dim]
V_S_t # [state_len, dim]


X_t # [img_len, img_dim] - image tokens, same deal
Q_X_t # [img_len, dim]
K_X_t # [img_len, dim]
V_X_t # [img_len, dim]


update(S_t-1, X_t):
	# basic rule
	S_t = S_t-1 - beta_t grad(S_t-1, X_t)

	# TTT3R formulation
	arg = Q_S_t-1 dot(dim) K_X_t # [state_len, img_len]
	arg = (1/img_len) * sum(img_len) arg # [state_len]
	beta_t = sigmoid(arg) 
	# explaination: act as a soft gate in gated attention
	
	# grad(S_t-1, X_t)
	arg = Q_S_t-1 dot(dim) K_X_t # [state_len, img_len]
	arg = softmax(img_len) arg 
	grad = -arg dot(img_len) V_X_t # [state_len, dim]
	
# Reading is the same


```

### Important code chunks

The reference implementation lives in the **TTT3R** repository (e.g. local clone at `MVA/INRIA/TTT3R` or [Inception3D/TTT3R](https://github.com/Inception3D/TTT3R)). The test-time behavior is **not** a second backward pass or optimizer step at inference; it reuses the **same** recurrent decoder as CUT3R to propose a next state `new_state_feat`, then **rescales** how much of that proposal is written into the persistent state using attention-derived gates—this is the code-side reading of “test-time training” for this project.

**Where to look**

| What | File (under repo root) |
|------|-------------------------|
| Flag `cut3r` vs `ttt3r` | `src/dust3r/model.py` — `ARCroco3DStereoConfig` (`model_update_type`) |
| State proposal + attention maps | `src/dust3r/model.py` — `_recurrent_rollout` → `_decoder` with `return_attn=True` |
| TTT3R gating + state write | `src/dust3r/model.py` — `_forward_impl` and the streaming forward path (two nearly identical blocks after each rollout) |
| CLI / eval | `demo.py`, `eval/*/launch.py` — `--model_update_type {cut3r,ttt3r}` sets `model.config.model_update_type` |

**Snippets from the TTT3R repo** (`src/` paths are relative to repo root)

Config flag on `ARCroco3DStereoConfig`:

```python
# src/dust3r/model.py — ARCroco3DStereoConfig.__init__
        pose_head=False,
        model_update_type="cut3r",
        **croco_kwargs,
    ):
        super().__init__()
        # ...
        self.model_update_type = model_update_type
```

Recurrent step: decoder proposes `new_state_feat` and (when `return_attn=True`) returns `cross_attn_state` for the TTT3R gate:

```python
# src/dust3r/model.py — _recurrent_rollout
    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        init_state_feat,
        img_mask=None,
        reset_mask=None,
        update=None,
        return_attn=False,
    ):
        (new_state_feat, dec), (self_attn_state, cross_attn_state, self_attn_img, cross_attn_img) = self._decoder(
            state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos, return_attn
        )
        new_state_feat = new_state_feat[-1]
        return new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img
```

Batch training forward (`_forward_impl`): first frame uses plain mask; later frames use **ttt3r** branch — rearrange attention, mean pool, **sigmoid** gate, then convex blend into `state_feat`:

```python
# src/dust3r/model.py — after each _recurrent_rollout in _forward_impl
            # update with learning rate
            if i  == 0:
                update_mask1 = update_mask
            else:
                if self.config.model_update_type == "cut3r":
                    update_mask1 = update_mask
                elif self.config.model_update_type == "ttt3r":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)') # [12, 16, 768, 1 + 576] -> [1, 768, 1 + 576, 12*16]
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    update_mask1 = update_mask * torch.sigmoid(state_query_img_key)[..., None] * 1.0
                else:
                    raise ValueError(f"Invalid model type: {self.config.model_update_type}")

            update_mask2 = update_mask
            state_feat = new_state_feat * update_mask1 + state_feat * (
                1 - update_mask1
            )  # update global state
            mem = new_mem * update_mask2 + mem * (
                1 - update_mask2
            )  # then update local state
```

Streaming / inference forward: same **ttt3r** logic, but the gate is skipped when `i == 0` **or** `reset_mask` (per-view reset):

```python
# src/dust3r/model.py — streaming forward loop
            # update with learning rate
            if i  == 0 or reset_mask:
                update_mask1 = update_mask
            else:
                if self.config.model_update_type == "cut3r":
                    update_mask1 = update_mask
                elif self.config.model_update_type == "ttt3r":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)') # [12, 16, 768, 1 + 576] -> [1, 768, 1 + 576, 12*16]
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    update_mask1 = update_mask * torch.sigmoid(state_query_img_key)[..., None] * 1.0
                else:
                    raise ValueError(f"Invalid model type: {self.config.model_update_type}")

            update_mask2 = update_mask
            state_feat = new_state_feat * update_mask1 + state_feat * (
                1 - update_mask1
            )  # update global state
            mem = new_mem * update_mask2 + mem * (
                1 - update_mask2
            )  # then update local state
```

CLI and wiring in `demo.py`:

```python
# demo.py
    parser.add_argument(
        "--model_update_type",
        type=str,
        default="cut3r",
        help="model update type: cut3r or ttt3r",
    )
    # ...
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.config.model_update_type = args.model_update_type
```

**What the code does**

1. **`_recurrent_rollout`** calls the decoder on `(state_feat, current image tokens, …)` and returns **`new_state_feat`** (the recurrent block’s output state) and **`cross_attn_state`**: cross-attention from **state tokens as query** to **image tokens as key/value**, per decoder layer and head. That is the same structural object as the pre-softmax $Q_{S_{t-1}} K_{X_t}^\top$ interaction in the TTT3R box above.

2. **CUT3R path:** the update mask `update_mask1` is just the image/update mask. The new state is blended in with  
   `state_feat = new_state_feat * update_mask1 + state_feat * (1 - update_mask1)`.

3. **TTT3R path:** `cross_attn_state` is concatenated over layers/heads and reshaped; a **scalar gate per state position** is built as  
   `state_query_img_key = cross_attn_state.mean(dim=(…))`  
   (mean over image positions and the merged layer/head dimension), then  
   `update_mask1 = update_mask * sigmoid(state_query_img_key)`.  
   The same convex combination as CUT3R is applied, but each state slot can take a **smaller step** toward `new_state_feat` when the sigmoid is small.

**Tie to the formulation above**

- **$\beta_t = \sigma(\cdot)$ and pooling:** the notes use a mean over the $Q_S K_X^\top$ logits along `img_len` before `sigmoid`. The code implements a related idea: **sigmoid** on a **summary statistic** of the state→image cross-attention map (mean over heads/layers and image tokens), producing a **soft gate** per state index—aligned with “$\beta_t$ acts as a soft gate in gated attention.”

- **$S_t = S_{t-1} - \beta_t \, \mathrm{grad}(\cdot)$:** the implementation does not instantiate a separate `grad` tensor or run autograd on a loss at test time. Instead, the decoder already defines a **forward** transition $S_{t-1} \mapsto \texttt{new\_state\_feat}$ (the CUT3R-style recurrent update inside the network). TTT3R **modulates the step size** toward that proposal via $\beta_t$ in the blend. So the “gradient” story in the math is an interpretation of the **direction and scale** of change encoded by the recurrent block; the **code** realizes the **$\beta_t$ gating** explicitly and keeps a single forward pass per frame (plus storing attention for the gate).

- **First frame / reset:** on the first timestep (or after a reset), the code skips the TTT3R gate and uses the plain `update_mask`, so behavior matches “initialize, then apply gated updates along the sequence.”

### Other notes