# Automatic differentiation

Each operation on a `Value` creates another `Value` connected to its operands.
`backward()` traverses this graph in reverse topological order and applies the
chain rule. Gradients in that graph are reset before each backward pass.

Supported differentiable operations include:

- Addition, subtraction, multiplication, and division
- `pow()` and `exp()`
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

For a list of logits, `softmax` returns normalized `double` probabilities.
`softmaxPick` randomly selects one of the original values using those
probabilities as weights.
