from time import sleep
from typing import Iterator


def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b


if __name__ == '__main__':
    for i in range(10):
        sleep(1)
        print(f'{i + 1} seconds passed...')
    fibonacci_sequence = fib(10)
    print('\n'.join(map(str, fibonacci_sequence)))
