# Getting started

## Installation

Add `dartgrad` to a Dart project:

```shell
dart pub add dartgrad
```

Import the package:

```dart
import 'package:dartgrad/dartgrad.dart';
```

## First expression

`Value` stores a scalar and records the operations used to produce new values.
Call `backward()` on the final value to populate the `grad` fields of every
value in its computation graph.

```dart
final a = Value(2.0);
final b = Value(3.0);
final result = (a * b + 1).tanh();

result.backward();

print(result.data);
print(a.grad);
print(b.grad);
```

Use `valueList` to convert a `List<num>` into a fixed-length `List<Value>`:

```dart
final input = [2.0, 3.0, -1.0].valueList;
```
