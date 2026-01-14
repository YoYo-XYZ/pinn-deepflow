
import time

# Start timing
start_time = time.time()

# Perform the operation 10,000 times
for _ in range(10000000):
    result = 100000 / 100

# End timing
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")
