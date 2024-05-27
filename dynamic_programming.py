# Imports:
# --------
from tqdm import tqdm
from time import time

# User definition:
# ----------------
n = 20

# Recursion:
# ----------
def fibonachi_series(n):
    if n == 1 or n == 2:
        result = 1
    else:
        result = fibonachi_series(n-1) + fibonachi_series(n-2)
    return result


start_rec = time()
for i in tqdm(range(1, n+1), desc="Fibonachi series without Memoization", colour="red"):
    fibonachi_series(n=i)
end_rec = time()
print("Time for recursion: ", (end_rec-start_rec), "\n")


# Memoization:
# ------------
def fibonachi_series_memo(n, memo):
    if memo[n] != None:
        return memo[n]
    elif n == 1 or n == 2:
        result = 1
    else:
        result = fibonachi_series(n-1) + fibonachi_series(n-2)
    memo[n] = result
    return result


memoization = dict(zip(range(1, n+1), [None]*n))

start_mem = time()
for i in tqdm(range(1, n+1), desc="Fibonachi series with Memoization", colour="green"):
    fibonachi_series_memo(n=i, memo=memoization)
end_mem = time()
print("Time for Memoization: ", (end_mem-start_mem), "\n")


# Bottom-up approach:
# -------------------
def fibonachi_bottom_up(n):
    if n == 1 or n == 2:
        return 1
    bottom_up = [1, 1]
    for i in range(3, n+1):
        bottom_up.append(bottom_up[i-2] + bottom_up[i-3])
    return bottom_up[n-1]


start_bottom = time()
for i in tqdm(range(1, n+1), desc="Fibonachi series Bottom Up", colour="blue"):
    fibonachi_bottom_up(n=i)
end_bottom = time()
print("Time for Bottom-Up: ", (end_bottom-start_bottom), "\n")
