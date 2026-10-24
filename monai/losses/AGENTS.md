# Losses

Shared conventions with `monai/metrics` (see `monai/metrics/AGENTS.md`):
most losses and metrics take an `include_background` flag (default `True`)
and a `reduction` from the enums in `monai.utils` (`LossReduction` here,
`MetricReduction` in metrics). Keep the flag names, defaults, and argument
order consistent with the neighboring classes when adding a new one.

## Things that are easy to get wrong

- Several losses (Dice and its variants in `dice.py`, focal loss) do not apply
  sigmoid or softmax internally unless `sigmoid=True` or `softmax=True` is
  passed. Check those flags before assuming the loss expects probabilities
  rather than logits, and do not change a default: it silently changes
  training results for existing users.
- `to_onehot_y` and `include_background` interact: with one-hot targets the
  background channel is channel 0, and excluding it slices after the one-hot
  conversion. Test both settings.
- Losses should support the batch and channel shapes their docstrings
  promise and return a scalar under the default reduction. Reductions other
  than the mean must not silently change the output shape contract.
- Numerical stability matters here (log of probabilities, division by sums
  over empty classes). Follow the epsilon handling of the closest existing
  loss instead of inventing a new one.
