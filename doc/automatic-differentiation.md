# Automatic differentiation

Each operation on a `Value` creates another `Value` connected to its operands.
`backward()` traverses this graph in reverse topological order and applies the
chain rule. Gradients in that graph are reset before each backward pass.

Supported differentiable operations include:

- Addition, subtraction, multiplication, and division
- `pow()`, `exp()`, and `log()`
- `tanh()`, `sigmoid()`, `relu()`, and `leakyRelu()`

```dart
final x = Value(2.0, label: 'x');
final y = Value(-3.0, label: 'y');
final output = (x * y + x).relu();

output.backward();

print(output.data);
print(x.grad);
print(y.grad);
```

Call `diagram()` to obtain a text representation of a computation graph, or
`show()` to print it. Labels make leaf values easier to identify.

For a list of logits, `softmax` returns normalized `Value` probabilities that
remain connected to the computation graph:

```dart
final logits = [1.0, 2.0, 3.0].valueList;
final probabilities = logits.softmax;

probabilities.last.backward();
print(logits.last.grad);
```

Use `argmax` or `argmin` to get the index of the largest or smallest value. For
random sampling, `weightedRandPick` treats the list as non-negative weights and
returns the selected index:

```dart
final predictedClass = logits.argmax;
final sampledClass = probabilities.weightedRandPick;
```

`weightedRandPick` does not apply softmax automatically. Its weights must be
finite and non-negative, and at least one must be greater than zero. Apply
`softmax` first when sampling from logits.
