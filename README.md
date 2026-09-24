# dartgrad

`dartgrad` is a small scalar-valued automatic differentiation and neural
network library for Dart. It is inspired by
[micrograd](https://github.com/karpathy/micrograd) and is intended for learning
how backpropagation and multilayer perceptrons work.

## Features

- Build dynamic computation graphs with `Value` objects.
- Run reverse-mode automatic differentiation with `backward()`.
- Use arithmetic, powers, exponentials, and common activation functions.
- Select `tanh`, `sigmoid`, `relu`, `leakyRelu`, or `linear` per layer.
- Calculate differentiable softmax probabilities and cross-entropy loss.
- Find minimum and maximum indexes or sample an index from weighted values.
- Create fully connected neural networks with `Neuron`, `Layer`, and `MLP`.
- Save trained MLP architectures, weights, and biases to JSON and load them.
- Render a computation graph as a text diagram.

## Getting started

Add the package to your project:

```shell
dart pub add dartgrad
```

Then import its public API:

```dart
import 'package:dartgrad/dartgrad.dart';
```

## Usage

Create a scalar expression and calculate its gradients:

```dart
final x = Value(2.0, label: 'x');
final y = (x * 3 + 2).pow(2);

y.backward();

print(y.data); // 64.0
print(x.grad); // 48.0
```

Build and train an MLP by updating its parameters:

```dart
final model = MLP(inputLength: 2, outputLengths: [4, 1]);
final input = [1.0, -2.0].valueList;
final target = Value(1.0);

for (var step = 0; step < 100; step++) {
  final prediction = model(input).single;
  final loss = (prediction - target).pow(2);

  loss.backward();
  for (final parameter in model.parameters) {
    parameter.data -= 0.1 * parameter.grad;
  }
}
```

Every layer defaults to `tanh`. Pass one activation per layer to customize the
model:

```dart
final model = MLP(
  inputLength: 2,
  outputLengths: [4, 1],
  activations: [Activation.relu, Activation.linear],
);
```

For multi-class classification, use a linear output layer and calculate
cross-entropy directly from its logits:

```dart
final classifier = MLP(
  inputLength: 784,
  outputLengths: [128, 10],
  activations: [Activation.relu, Activation.linear],
);

final logits = classifier(image.valueList);
final loss = logits.crossEntropy(targetClass: label);
loss.backward();

final probabilities = logits.softmax;
final predictedClass = logits.argmax;
final sampledClass = probabilities.weightedRandPick;
```

Save the trained model and load it again later:

```dart
await model.save('model.json');

final loadedModel = await MLP.load('model.json');
final prediction = loadedModel(input).single;
```

See [`example/dartgrad_example.dart`](example/dartgrad_example.dart) for
single-output and multi-output training examples.

## Documentation

- [Documentation index](doc/README.md)
- [Getting started](doc/getting-started.md)
- [Automatic differentiation](doc/automatic-differentiation.md)
- [Neural networks](doc/neural-networks.md)

This package is designed for education and experimentation rather than
production machine-learning workloads. Issues and contributions are welcome in
the [GitHub repository](https://github.com/Descrout/dartgrad).
