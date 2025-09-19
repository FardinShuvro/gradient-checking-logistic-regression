import math
import random
import time


def sigmoid(z):
    """Compute the logistic sigmoid function."""
    return 1 / (1 + math.exp(-z))


def dot_product(vector1, vector2):
    """Manual dot product implementation."""
    return sum(x * y for x, y in zip(vector1, vector2))


def hypothesis(X_row, theta):
    """Compute hypothesis h_theta(x) = sigma(theta^T x)."""
    z = dot_product(X_row, theta)
    return sigmoid(z)


def cost_function(X, y, theta):
    """Compute the logistic regression cost function."""
    m = len(y)
    total_cost = 0.0

    for i in range(m):
        h = hypothesis(X[i], theta)
        # Add small value to avoid log(0)
        if y[i] == 1:
            total_cost += -math.log(h + 1e-10)
        else:
            total_cost += -math.log(1 - h + 1e-10)

    return total_cost / m


def analytical_gradient(X, y, theta):
    """Compute analytical gradient of the cost function."""
    m = len(y)
    n = len(theta)
    grad = [0.0] * n

    for i in range(m):
        h = hypothesis(X[i], theta)
        error = h - y[i]

        for j in range(n):
            grad[j] += error * X[i][j]

    # Divide by m
    for j in range(n):
        grad[j] /= m

    return grad


def numerical_gradient(X, y, theta, epsilon):
    """Compute numerical gradient using centered finite differences."""
    n = len(theta)
    num_grad = [0.0] * n

    for i in range(n):
        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon

        cost_plus = cost_function(X, y, theta_plus)
        cost_minus = cost_function(X, y, theta_minus)

        num_grad[i] = (cost_plus - cost_minus) / (2 * epsilon)

    return num_grad


def vector_norm(vector):
    """Compute Euclidean norm of a vector."""
    return math.sqrt(sum(x * x for x in vector))


def generate_dataset(m, n):
    """Generate synthetic dataset for testing."""
    random.seed(int(time.time()))

    # True parameters for generating data
    true_theta = [0.5] + [random.uniform(-1, 1) for _ in range(n - 1)]

    # Generate dataset
    X = []
    y = []

    for _ in range(m):
        # Create features (first element is bias term = 1)
        features = [1.0] + [random.uniform(-1, 1) for _ in range(n - 1)]

        # Compute probability
        z = dot_product(features, true_theta)
        probability = sigmoid(z)

        
        label = 1.0 if random.random() < probability else 0.0

        X.append(features)
        y.append(label)

    return X, y, true_theta


def gradient_check(X, y, theta, epsilon, tolerance):
    """Perform gradient checking and return results."""
    # Compute both gradients
    anal_grad = analytical_gradient(X, y, theta)
    num_grad = numerical_gradient(X, y, theta, epsilon)

    # Calculate relative differences
    relative_diffs = []
    for i in range(len(anal_grad)):
        diff = abs(anal_grad[i] - num_grad[i])
        avg = (abs(anal_grad[i]) + abs(num_grad[i])) / 2
        if avg > 1e-10:
            relative_diffs.append(diff / avg)
        else:
            relative_diffs.append(diff)

    # Compute overall difference metric
    diff_vector = [a - n for a, n in zip(anal_grad, num_grad)]
    norm_diff = vector_norm(diff_vector)
    norm_anal = vector_norm(anal_grad)
    norm_num = vector_norm(num_grad)
    overall_diff = norm_diff / (norm_anal + norm_num + 1e-10)

    # Check if gradients agree within tolerance
    passes = overall_diff < tolerance

    return anal_grad, num_grad, relative_diffs, overall_diff, passes


# [The rest of your functions (print_parameters, print_results, get_user_input, main)
# remain exactly the same, just remove any np. prefixes]

def print_parameters(m, n, epsilon, tolerance, theta):
    """Print parameters with detailed explanations."""
    print("\n" + "=" * 60)
    print("PARAMETER EXPLANATION")
    print("=" * 60)

    print(f"\nEpsilon (ε = {epsilon:.1e}): Step size for finite differences")
    print("  - Used to perturb parameters for numerical gradient calculation")
    print("  -  Too large: Poor approximation of true derivative")
    print("  -  Too small: Numerical precision issues")
    print("  -  Recommended: 1e-4 to 1e-6")

    print(f"\nTolerance ({tolerance:.1e}): Maximum allowed difference between gradients")
    print("  - Determines if gradient check passes or fails")
    print("  -  Typical values: 1e-7 to 1e-9")
    print("  -  Difference < tolerance: Implementation is correct ✓")
    print("  -  Difference > tolerance: Potential bug in gradient code ✗")

    print(f"\nTraining examples (m): {m}")
    print(f"Features including bias (n): {n}")
    print(f"Initial theta: {theta}")


def print_results(anal_grad, num_grad, relative_diffs, overall_diff, passes, tolerance):
    """Print gradient comparison results."""
    print("\n" + "=" * 60)
    print("GRADIENT COMPARISON RESULTS")
    print("=" * 60)

    print(f"\n{'Parameter':<12} {'Analytical':<15} {'Numerical':<15} {'Relative Diff':<15}")
    print("-" * 60)

    for i in range(len(anal_grad)):
        param_name = f"theta[{i}]" if i > 0 else "bias"
        print(f"{param_name:<12} {anal_grad[i]:<15.8f} {num_grad[i]:<15.8f} {relative_diffs[i]:<15.2e}")

    print("\n" + "=" * 40)
    print(f"Overall Difference Metric: {overall_diff:.2e}")
    print(f"Tolerance Threshold: {tolerance:.0e}")
    print("=" * 40)

    if passes:
        print(" GRADIENT CHECK PASSED!")
        print(" Your analytical gradient implementation appears correct!")
        print(" You can confidently use analytical gradients for training.")
    else:
        print(" GRADIENT CHECK FAILED!")
        print(" There may be a bug in your analytical gradient code.")
        print("   Please check your derivative calculations and implementation.")


def get_user_input():
    """Get parameters from user with validation."""
    print("=== Gradient Checking for Logistic Regression ===")
    print("Press Enter to use default values\n")

    try:
        # Get number of training examples
        m_input = input(f"Enter number of training examples (m) [default: 50]: ").strip()
        m = int(m_input) if m_input else 50
        m = max(10, m)  # Ensure at least 10 examples

        # Get number of features (including bias)
        n_input = input(f"Enter number of features including bias (n) [default: 3]: ").strip()
        n = int(n_input) if n_input else 3
        n = max(2, n)  # Ensure at least bias + 1 feature

        # Get epsilon
        epsilon_input = input(f"Enter epsilon (step size) [default: 1e-4]: ").strip()
        epsilon = float(epsilon_input) if epsilon_input else 1e-4
        epsilon = max(1e-10, min(epsilon, 1e-1))

        # Get tolerance
        tolerance_input = input(f"Enter tolerance [default: 1e-7]: ").strip()
        tolerance = float(tolerance_input) if tolerance_input else 1e-7
        tolerance = max(1e-12, min(tolerance, 1e-3))

        # Get initial theta values
        print(f"\nEnter initial parameter values for theta ({n} values):")
        print("Note: theta[0] is the bias term")

        theta = []
        for i in range(n):
            default_value = 0.1 if i == 0 else 0.0
            prompt = f"theta[{i}] [default: {default_value:.3f}]: "
            value_input = input(prompt).strip()

            try:
                value = float(value_input) if value_input else default_value
            except ValueError:
                value = default_value

            theta.append(value)

        return m, n, epsilon, tolerance, theta

    except ValueError:
        print("Invalid input! Using default values.")
        return 50, 3, 1e-4, 1e-7, [0.1, -0.2, 0.05]


def main():
    """Main function to run gradient checking."""
    # Get user input (using smaller defaults for performance)
    m, n, epsilon, tolerance, theta = get_user_input()

    # Generate synthetic dataset
    print(f"\nGenerating synthetic dataset with {m} examples and {n} features...")
    X, y, true_theta = generate_dataset(m, n)

    # Print parameters with explanations
    print_parameters(m, n, epsilon, tolerance, theta)

    # Display initial cost
    initial_cost = cost_function(X, y, theta)
    print(f"\nInitial cost: {initial_cost:.6f}")

    # Perform gradient checking
    print(f"\nPerforming gradient check...")
    print(f"This will require {2 * n * m} cost function evaluations...")

    start_time = time.time()
    results = gradient_check(X, y, theta, epsilon, tolerance)
    end_time = time.time()

    anal_grad, num_grad, relative_diffs, overall_diff, passes = results

    # Print results
    print_results(anal_grad, num_grad, relative_diffs, overall_diff, passes, tolerance)

    # Display performance information
    computation_time = end_time - start_time
    print(f"\nComputation time: {computation_time:.3f} seconds")
    print(f"Cost function evaluations: {2 * n * m}")


if __name__ == "__main__":

    main()
