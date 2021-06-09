def pop_and_append(x):
  x.append(x.pop(0))
  return x[-1]