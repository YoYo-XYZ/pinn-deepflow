
import time

# Create a large list of numbers
numbers = list(range(10))

# Benchmark sum()
start = time.time()
total_sum = sum(numbers)
end = time.time()
print(f"sum() took: {end - start:.6f} seconds")

# Benchmark manual addition using iteration
start = time.time()
manual_sum = 0
for num in numbers:
    manual_sum += num
end = time.time()
print(f"Manual iteration took: {end - start:.6f} seconds")
