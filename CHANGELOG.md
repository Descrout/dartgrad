## 1.0.2

- Added per-layer activation configuration to `MLP` with `tanh` defaults.
- Added `tanh`, `sigmoid`, `relu`, `leakyRelu`, and `linear` activation options.
- Added Xavier initialization for tanh, sigmoid, and linear layers, and He
  initialization for ReLU layers. Biases now start at zero.
- Updated model persistence to store and restore each layer's activation while
  retaining support for version 1 model files.
- Made `List<Value>.softmax` differentiable by returning `List<Value>`.
- Added numerically stable `crossEntropy()` loss for multi-class
  classification.
- Added differentiable `Value.log()` support.
- Updated tests and documentation for activation configuration, model
  persistence, softmax, and cross-entropy.
- Breaking: replaced the single `MLP` `activation` argument with the
  per-layer `activations` list.
- Breaking: changed `List<Value>.softmax` from `List<double>` to `List<Value>`.

## 1.0.1

- Added `load` and `save` functionality.

## 1.0.0

- Initial version.
