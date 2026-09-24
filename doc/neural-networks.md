# Neural networks

`dartgrad` provides three fully connected building blocks:

- `Neuron` computes a weighted sum plus bias and applies an activation.
- `Layer` evaluates multiple neurons for the same input.
- `MLP` chains layers and exposes all trainable `parameters`.

The following model accepts three values, has two hidden layers with four
neurons each, and produces one output:

```dart
final model = MLP(inputLength: 3, outputLengths: [4, 4, 1]);
final prediction = model([2.0, 3.0, -1.0].valueList).single;
```

Each layer defaults to `Activation.tanh`. Pass an activation for every entry in
`outputLengths` to configure layers independently:

```dart
final model = MLP(
  inputLength: 3,
  outputLengths: [4, 4, 1],
  activations: [
    Activation.relu,
    Activation.relu,
    Activation.linear,
  ],
);
```

Available values are `tanh`, `sigmoid`, `relu`, `leakyRelu`, and `linear`.
The number of activations must match the number of layers.

Training consists of a forward pass, a backward pass, and a parameter update:

```dart
final target = Value(1.0);

for (var step = 0; step < 100; step++) {
  final prediction = model([2.0, 3.0, -1.0].valueList).single;
  final loss = (prediction - target).pow(2);

  loss.backward();
  for (final parameter in model.parameters) {
    parameter.data -= 0.1 * parameter.grad;
  }
}
```

Each selected activation is used by every neuron in its layer. Inputs must
match the model's `inputLength`; outputs follow the final value in
`outputLengths`.

Weights use zero-centered, activation-aware initialization. Tanh, sigmoid, and
linear layers use Xavier uniform initialization; ReLU and leaky ReLU layers use
He uniform initialization. Biases start at zero. This scaling reduces exploding
or vanishing activations compared with using one fixed range for every layer.

## Multi-class classification

Use `Activation.linear` for the final layer and pass its raw outputs, or logits,
to `crossEntropy()`. The target is the zero-based index of the correct class:

```dart
final model = MLP(
  inputLength: 784,
  outputLengths: [128, 10],
  activations: [Activation.relu, Activation.linear],
);

final logits = model(image.valueList);
final loss = logits.crossEntropy(targetClass: label);

loss.backward();
```

`crossEntropy()` uses a numerically stable log-sum-exp calculation and returns a
`Value`, so gradients flow directly into the model. Use `logits.softmax` when
normalized class probabilities are needed. It returns `List<Value>` and also
remains differentiable. Softmax does not need to be configured as a per-neuron
activation because it operates on the complete output layer.

## Saving and loading

Save a trained model to a JSON file with `save()`:

```dart
await model.save('model.json');
```

Restore its architecture, weights, and biases with `MLP.load()`:

```dart
final model = await MLP.load('model.json');
final prediction = model([2.0, 3.0, -1.0].valueList).single;
```

The file includes every layer's activation and a format version so future
formats can be distinguished. Gradients are not persisted because they are
temporary values produced during backpropagation. File persistence uses
`dart:io` and is intended for Dart VM platforms.
