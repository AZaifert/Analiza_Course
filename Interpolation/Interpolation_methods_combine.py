import numpy as np
#code was refactored on the original code that was provided in the course together with the samples of pseudocode.
#rewriting was made with the use of copilot and some hand made changes to make the code more readable and understandable.
#exxesive comments were removed and the code was made more compact.
# ============================================================
# 1. lagrange_interpolation
# ============================================================
def lagrange_interpolation(x_data, y_data, x):

    n = len(x_data)
    result = 0.0
    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term
    return result

# ============================================================
# 2. neville_interpolation
# ============================================================
def neville(x_data, y_data, x):

    n = len(x_data)
    # Initialize the Neville tableau
    tableau = [[0.0]*n for _ in range(n)]
    for i in range(n):
        tableau[i][0] = y_data[i]

    for j in range(1, n):
        for i in range(n - j):
            numerator = ((x - x_data[i + j]) * tableau[i][j - 1] -
                         (x - x_data[i]) * tableau[i + 1][j - 1])
            denominator = (x_data[i] - x_data[i + j])
            tableau[i][j] = numerator / denominator

    return tableau[0][n - 1]

# ============================================================
# 3. linear_interpolation
# ============================================================
def linear_interpolation(points, x):

    # Sort points by their x-value
    points = sorted(points, key=lambda p: p[0])

    # Check intervals
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if x1 <= x <= x2:
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    # Extrapolate if outside range
    if x < points[0][0]:
        x1, y1 = points[0]
        x2, y2 = points[1]
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    else:
        x1, y1 = points[-2]
        x2, y2 = points[-1]
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

# ============================================================
# 4. polynomial_interpolation
# ============================================================
def polynomial_interpolation(points, x):

    n = len(points)
    X = np.zeros((n, n))
    Y = np.zeros(n)

    for i, (xi, yi) in enumerate(points):
        for j in range(n):
            X[i, j] = xi**j
        Y[i] = yi

    # Solve for polynomial coefficients
    coeffs = np.linalg.solve(X, Y)

    # Evaluate polynomial at x
    return sum(coeffs[j] * x**j for j in range(n))

# ============================================================
# 5. cubic_spline_interpolation
# ============================================================
def cubic_spline_interpolation(x_data, y_data, x):

    # Sort data if not sorted
    xy = sorted(zip(x_data, y_data), key=lambda p: p[0])
    x_data = [p[0] for p in xy]
    y_data = [p[1] for p in xy]

    n = len(x_data)
    h = [x_data[i+1] - x_data[i] for i in range(n-1)]

    # Compute alpha for the system
    alpha = [0]*(n-1)
    for i in range(1, n-1):
        alpha[i] = (3/h[i]) * (y_data[i+1] - y_data[i]) \
                   - (3/h[i-1]) * (y_data[i] - y_data[i-1])

    # Tridiagonal system arrays
    l = [1]*n
    mu = [0]*n
    z = [0]*n

    for i in range(1, n-1):
        l[i] = 2*(x_data[i+1] - x_data[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]

    # Back-substitution
    c = [0]*n
    b = [0]*(n-1)
    d = [0]*(n-1)

    for j in range(n-2, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = ((y_data[j+1] - y_data[j]) / h[j]) - h[j]*(c[j+1] + 2*c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3*h[j])

    # Find interval for x
    i = 0
    if x < x_data[0]:
        i = 0
    elif x > x_data[-1]:
        i = n-2
    else:
        for j in range(n-1):
            if x_data[j] <= x <= x_data[j+1]:
                i = j
                break

    dx = x - x_data[i]
    # Spline polynomial
    return y_data[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

# ============================================================
# main
# ============================================================
def main():
    # Example data
    x_data = [0, 1, 2, 3, 4]
    y_data = [0, 0.84, 0.91, 0.14, -0.76]
    x_test = 1.5

    # Convert data to (x, y) tuples for certain routines
    points = list(zip(x_data, y_data))

    # User menu
    print("Choose an interpolation method:\n"
          "1) Lagrange\n"
          "2) Neville\n"
          "3) Linear\n"
          "4) Polynomial\n"
          "5) Cubic Spline")
    choice = input("Enter choice [1-5]: ")

    if choice == '1':
        result = lagrange_interpolation(x_data, y_data, x_test)
        print(f"[Lagrange]  Value at x={x_test} is {result}")
    elif choice == '2':
        result = neville(x_data, y_data, x_test)
        print(f"[Neville]   Value at x={x_test} is {result}")
    elif choice == '3':
        result = linear_interpolation(points, x_test)
        print(f"[Linear]    Value at x={x_test} is {result}")
    elif choice == '4':
        result = polynomial_interpolation(points, x_test)
        print(f"[Polynomial] Value at x={x_test} is {result}")
    elif choice == '5':
        result = cubic_spline_interpolation(x_data, y_data, x_test)
        print(f"[Cubic Spline] Value at x={x_test} is {result}")
    else:
        print("Invalid choice. Please run again.")

if __name__ == '__main__':
    main()
