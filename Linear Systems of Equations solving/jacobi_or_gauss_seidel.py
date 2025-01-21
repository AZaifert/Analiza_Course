import numpy as np

def is_diagonally_dominant(a):
    n = a.shape[0]
    for i in range(n):
        row_sum = np.sum(np.abs(a[i, :])) - abs(a[i, i])
        if abs(a[i, i]) < row_sum:
            return False
    return True

def pivot_for_diagonal_dominance(a, b):
    a, b = a.copy(), b.copy()
    n = a.shape[0]
    for i in range(n):
        max_row = np.argmax(np.abs(a[i:, i])) + i
        if max_row != i:
            a[[i, max_row]], b[[i, max_row]] = a[[max_row, i]], b[[max_row, i]]
    return a, b

def jacobi_iteration(a, b, x_current):
    n = a.shape[0]
    x_new = np.copy(x_current)
    for i in range(n):
        s = np.sum(a[i, :] * x_current) - a[i, i] * x_current[i]
        x_new[i] = (b[i] - s) / a[i, i]
    return x_new

def gauss_seidel_iteration(a, b, x_current):
    n = a.shape[0]
    x_new = np.copy(x_current)
    for i in range(n):
        s = sum(a[i, j] * x_new[j] for j in range(n) if j != i)
        x_new[i] = (b[i] - s) / a[i, i]
    return x_new

def jacobi_solver(a, b, epsilon=1e-4, max_iter=100):
    print("=== Jacobi Method ===")
    if not is_diagonally_dominant(a):
        print("Matrix is not diagonally dominant. Attempting to reorder...")
        a, b = pivot_for_diagonal_dominance(a, b)
        if not is_diagonally_dominant(a):
            print("Still not diagonally dominant after row swaps.")
    x = np.zeros_like(b)
    for k in range(1, max_iter + 1):
        x_new = jacobi_iteration(a, b, x)
        print(f"Iteration {k}: x = {x_new}")
        if np.linalg.norm(x_new - x, ord=np.inf) < epsilon:
            print(f"Converged in {k} iterations.")
            return x_new
        x = x_new
    print("Jacobi method did NOT converge within the maximum number of iterations.")
    return None

def gauss_seidel_solver(a, b, epsilon=1e-4, max_iter=100):
    print("=== Gauss-Seidel Method ===")
    if not is_diagonally_dominant(a):
        print("Matrix is not diagonally dominant. Attempting to reorder...")
        a, b = pivot_for_diagonal_dominance(a, b)
        if not is_diagonally_dominant(a):
            print("Still not diagonally dominant after row swaps.")
    x = np.zeros_like(b)
    for k in range(1, max_iter + 1):
        x_new = gauss_seidel_iteration(a, b, x)
        print(f"Iteration {k}: x = {x_new}")
        if np.linalg.norm(x_new - x, ord=np.inf) < epsilon:
            print(f"Converged in {k} iterations.")
            return x_new
        x = x_new
    print("Gauss-Seidel method did NOT converge within the maximum number of iterations.")
    return None

def main():
    A = np.array([
        [4, 2, 0],
        [2, 10, 4],
        [0, 4, 5]
    ], dtype=float)
    b = np.array([2, 6, 5], dtype=float)

    print("Choose the method you want to use:")
    print("1) Jacobi")
    print("2) Gauss-Seidel")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == '1':
        x_solution = jacobi_solver(A, b, epsilon=1e-4, max_iter=100)
        if x_solution is not None:
            print("Solution (Jacobi):", x_solution)
    elif choice == '2':
        x_solution = gauss_seidel_solver(A, b, epsilon=1e-4, max_iter=100)
        if x_solution is not None:
            print("Solution (Gauss-Seidel):", x_solution)
    else:
        print("Invalid choice. Please enter 1 for Jacobi or 2 for Gauss-Seidel.")

if __name__ == "__main__":
    main()