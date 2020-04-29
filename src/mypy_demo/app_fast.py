def fib(n):
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b


if __name__ == '__main__':
    fibonacci_sequence = fib('ten')
    print('\n'.join(map(str, fibonacci_sequence)))
