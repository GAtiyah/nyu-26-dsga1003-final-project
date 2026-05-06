# Batch Scale Explanation

In the residual scaling hook, `layer_scales` can have two shapes.

## Prompt Mean Pooling

For prompt-conditioned models, the prompt is first encoded by the frozen Qwen model. The resulting prompt hidden states have shape:

```text
[batch_size, prompt_length, hidden_size]
```

For example:

```text
[6, 96, 1024]
```

The code then averages the prompt token hidden states:

```python
mean_pooled_hidden = hidden_states.mean(dim=1)
```

This compresses the 96 token vectors for each input example into one vector:

```text
[6, 96, 1024] -> [6, 1024]
```

If an `attention_mask` is provided, the average is computed only over valid tokens:

```python
weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
weighted_sum = (hidden_states * weights).sum(dim=1)
token_count = weights.sum(dim=1).clamp_min(1.0)
mean_pooled_hidden = weighted_sum / token_count
```

This means padding tokens do not affect the average.

The reason for this pooling step is that the controller needs one set of layer scales per input example, not one set per token. Pooling converts the prompt sequence representation into a fixed-size prompt summary that the MLP can use to output layer scales.

The model also uses the final prompt token hidden state:

```python
last_token_hidden = hidden_states[:, -1, :]
```

Then it concatenates both summaries:

```python
torch.cat([last_token_hidden, mean_pooled_hidden], dim=-1)
```

If `hidden_size = 1024`, this produces:

```text
[batch_size, 2048]
```

This is why the prompt-conditioned MLP input size is `2 * hidden_size`.

## Why Use Both Last Token and Mean Pooling

`last_token_hidden` is the hidden state at the final prompt token position. In a causal language model, this final position can attend to previous prompt tokens, so it often contains a strong contextual summary of the prompt.

However, it can also be strongly influenced by the final token itself, such as punctuation, newline tokens, or special tokens. It compresses the full prompt into one position.

`mean_pooled_hidden` averages the hidden states of all valid prompt tokens. It gives every prompt token a direct contribution to the summary, which can make it more stable for capturing the prompt's general topic, style, or domain.

The model concatenates both:

```python
torch.cat([last_token_hidden, mean_pooled_hidden], dim=-1)
```

This gives the controller two complementary summaries:

```text
last_token_hidden: final-position contextual state
mean_pooled_hidden: global average prompt state
```

Using only `last_token_hidden` could work, but adding mean pooling gives the MLP extra global prompt information while keeping the controller lightweight.

## Static Scale

```python
if layer_scales.dim() == 1:
    scale = layer_scales[layer_index].view(1, 1, 1)
```

Here `layer_scales` has shape:

```text
[num_layers]
```

This means each layer has one fixed scale, and every input in the batch uses the same scale for that layer.

After selecting the current layer's scale, `.view(1, 1, 1)` reshapes it so it can broadcast over:

```text
[batch_size, sequence_length, hidden_size]
```

So the same scale is applied to every example, every token, and every hidden dimension.

## Prompt-Conditioned Scale

```python
else:
    scale = layer_scales[:, layer_index].view(layer_scales.shape[0], 1, 1)
```

Here `layer_scales` has shape:

```text
[batch_size, num_layers]
```

This means each input example in the batch has its own scale for each layer.

`layer_scales[:, layer_index]` selects the current layer's scale for every example in the batch. Its shape is:

```text
[batch_size]
```

Then `.view(layer_scales.shape[0], 1, 1)` reshapes it to:

```text
[batch_size, 1, 1]
```

This lets each example use its own scale, while sharing that scale across all tokens and hidden dimensions for that example.

In short:

```text
static: one scale for the whole batch at each layer
prompt-conditioned: one scale per input example at each layer
```

## Static vs Prompt-Conditioned Training Target

For the static write-strength model, the model directly trains one fixed scaler per Qwen layer.

If Qwen has 28 layers, the trainable scaler vector has shape:

```text
[28]
```

This single `[28]` vector is learned from the training set and then shared by all input examples. It is not prompt-specific.

For the prompt-conditioned write-strength model, the model does not directly train one fixed `[28]` vector. Instead, it trains an MLP:

```python
raw_scales = self.conditioner(prompt_representation)
```

The MLP takes a prompt representation as input and outputs one scaler per layer for that specific prompt.

For one input example:

```text
prompt representation -> MLP -> [28]
```

For a batch:

```text
[batch_size, 2 * hidden_size] -> MLP -> [batch_size, 28]
```

So:

```text
static: trains 28 fixed scalers
prompt-conditioned: trains an MLP that generates 28 scalers from each prompt
```
