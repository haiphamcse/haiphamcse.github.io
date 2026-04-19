---
title: "Cambrian-S Stage 1: alignment warmup and mm_projector"
date: 2026-04-19
slug: vsi-bench-cambrian-s
description: "Vision-language alignment stage in cambrian-s: what is trained, data format, and pseudo-code."
---

### Cambrian-S architecture

This note describes what **Stage 1** does in the `cambrian-s` repo (script: `cambrian/scripts/cambrians_7b_s1.sh`) and gives a practical pseudo-code training loop.

Stage 1 is the **vision-language alignment warmup** stage.

- **Goal**: make the LLM understand visual tokens from the vision encoder through the multimodal connector (projector), before large-scale image/video instruction tuning.
- **Backbone**: `Qwen/Qwen2.5-7B-Instruct`.
- **Vision encoder**: `google/siglip2-so400m-patch14-384`.
- **Data**: `Cambrian-Alignment` JSONL + image folder.

### What is trained vs frozen

From `cambrians_7b_s1.sh`:

- `--connector_only True`
- `--tune_mm_mlp_adapter True`
- `--unfreeze_mm_vision_tower False`
- `--mm_projector_lr 1e-3`
- `--learning_rate 1e-3`

**Interpretation**

- Training focuses on the **multimodal projector / adapter path** (fast alignment).
- The vision tower stays frozen.
- The language model is used as a fixed foundation while the visual connector is aligned.

### Input format assumptions in Stage 1

The data loader in `train_fsdp.py` expects:

- one JSON object per line (`.jsonl`)
- `conversations`: list of alternating `human` / `gpt` turns
- image sample has an `image` field (relative path under `--image_folder`)
- if image token is missing, loader may inject `<image>` into first human turn
- a sample cannot contain both `image` and `video`

Minimal sample shape:

```json
{
  "image": "subdir/sample.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe the scene."},
    {"from": "gpt", "value": "A person is walking in a park."}
  ]
}
```

### Stage 1 training loop (pseudo-code)

```python
# ---------------------------------------------------------
# Stage 1: vision-language alignment (projector warmup)
# ---------------------------------------------------------

def run_stage1_training(config):
    # 1) Setup distributed XLA/FSDP runtime
    initialize_xla_runtime()
    set_random_seed(config.seed)

    # 2) Build tokenizer + multimodal model
    tokenizer = build_tokenizer(config.model_name_or_path, version="qwen_2")
    model = build_cambrian_model(
        llm_name=config.model_name_or_path,  # Qwen2.5-7B-Instruct
        vision_tower_name="google/siglip2-so400m-patch14-384",
        mm_projector_type="mlp2x_gelu",
        mm_use_im_newline_token=True,
    )

    # 3) Freeze / unfreeze according to Stage 1 policy
    freeze_module(model.language_model)  # keep LLM fixed for alignment warmup
    freeze_module(model.vision_tower)  # script sets unfreeze_mm_vision_tower=False
    unfreeze_module(model.mm_projector)  # script tunes adapter/projector

    # 4) Build dataset + dataloader
    train_dataset = LazySupervisedDataset(
        data_path=config.data_path,  # Cambrian-Alignment .jsonl
        image_folder=config.image_folder,
        image_aspect_ratio="pad",
        max_images_per_sample=1,
    )
    train_loader = make_dataloader(
        dataset=train_dataset,
        batch_size=config.per_device_train_batch_size,
        num_workers=config.dataloader_num_workers,
        shuffle=True,
        collate_fn=multimodal_collator,
    )

    # 5) Optimizer/scheduler with Stage 1 LR
    optimizer = AdamW(
        params=trainable_parameters(model),  # mostly projector parameters
        lr=1e-3,
        weight_decay=0.0,
    )
    scheduler = cosine_scheduler(
        optimizer=optimizer,
        warmup_ratio=0.06,
        total_steps=compute_total_steps(train_loader, epochs=1),
    )

    # 6) Wrap model with XLA FSDP and enable gradient checkpointing
    model = wrap_with_xla_fsdp(model, fsdp_config_path="fsdp_config.json")
    enable_gradient_checkpointing(model)
    model.train()

    global_step = 0
    for epoch in range(1):  # script uses num_train_epochs=1
        for batch in train_loader:
            # batch contains:
            # input_ids, labels, attention_mask, images, image token indices...
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if global_step % config.logging_steps == 0:
                log_metrics({"loss": loss.item(), "lr": scheduler.current_lr})

            if global_step % config.save_steps == 0 and global_step > 0:
                save_checkpoint_to_output_dir(model, optimizer, scheduler, step=global_step)

            global_step += 1

    save_final_checkpoint(model, optimizer, scheduler)
    return model
```

### Why this stage matters

S1 reduces optimization instability in later stages by first teaching the connector how to map visual features into the LLM token space. Then S2/S3/S4 can scale instruction tuning (image → video → spatial video) from a better-initialized multimodal interface.

### What `mm_projector` is used for

`mm_projector` is the **feature-space bridge** between vision tokens and LLM tokens.

- Vision encoder outputs features in vision hidden size (here around 1152 from SigLIP2).
- LLM expects embeddings in its own hidden size (Qwen hidden size).
- `mm_projector` (here `mlp2x_gelu`) maps visual features to the LLM embedding space.
- The projected visual tokens are inserted at `<image>` token positions (and related image/video token slots) before passing to LLM decoder layers.

In Stage 1, this mapping is the main thing being learned, so the model learns: “given visual features, produce token embeddings that the frozen LLM can already reason over.”

### Pseudo-code: how `mm_projector` is used in forward

```python
def forward_multimodal(
    input_ids,
    attention_mask,
    labels,
    images,  # [B, N_img, 3, H, W] or packed frame/image tensors
    image_token_indices,  # positions where visual tokens should be inserted
):
    # 1) Text path: get token embeddings from the LLM embedding table
    text_embeds = llm.embed_tokens(input_ids)  # [B, T, D_llm]

    # 2) Vision path: encode images/frames (usually frozen in Stage 1)
    with torch.no_grad():  # because unfreeze_mm_vision_tower=False in S1
        vision_feats = vision_tower(images)  # [B, N_vis, D_vis]

    # 3) Project vision features into LLM embedding space
    #    This is the key role of mm_projector.
    projected_vis_embeds = mm_projector(vision_feats)  # [B, N_vis, D_llm]

    # 4) Merge: replace/insert at multimodal token positions
    fused_embeds = text_embeds.clone()
    fused_embeds = scatter_visual_embeddings(
        fused_embeds,
        projected_vis_embeds,
        image_token_indices,
    )  # still [B, T, D_llm]

    # 5) Run LLM decoder with fused embeddings
    outputs = llm(
        inputs_embeds=fused_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )
    return outputs
```

### Pseudo-code: why Stage 1 optimizes `mm_projector`

```python
def stage1_step(batch):
    # freeze language model + vision tower
    set_requires_grad(llm, False)
    set_requires_grad(vision_tower, False)

    # train projector
    set_requires_grad(mm_projector, True)

    outputs = forward_multimodal(**batch)
    loss = outputs.loss  # autoregressive language modeling loss
    loss.backward()  # gradients mostly flow into mm_projector
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```
