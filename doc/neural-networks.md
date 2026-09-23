# Neural networks

`dartgrad` provides three fully connected building blocks:

- `Neuron` computes a weighted sum plus bias and applies `tanh`.
- `Layer` evaluates multiple neurons for the same input.
- `MLP` chains layers and exposes all trainable `parameters`.

The following model accepts three values, has two hidden layers with four
neurons each, and produces one output:

```dart
final model = MLP(inputLength: 3, outputLengths: [4, 4, 1]);
final prediction = model([2.0, 3.0, -1.0].valueList).single;
```

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

Every neuron currently uses `tanh`, including neurons in the final layer.
Inputs must match the model's `inputLength`; outputs follow the final value in
`outputLengths`.

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

The file includes a format version so future formats can be distinguished.
Gradients are not persisted because they are temporary values produced during
backpropagation. File persistence uses `dart:io` and is intended for Dart VM
platforms.
