#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Default values
#define DEFAULT_M 100
#define DEFAULT_N 5
#define DEFAULT_EPSILON 1e-4
#define DEFAULT_TOLERANCE 1e-7

// Function prototypes
double sigmoid(double z);
double hypothesis(double *theta, double *x, int n_features);
double cost_function(double **X, double *y, double *theta, int m, int n);
void analytical_gradient(double **X, double *y, double *theta, double *grad, int m, int n);
void numerical_gradient(double **X, double *y, double *theta, double *num_grad, int m, int n, double epsilon);
void generate_dataset(double **X, double *y, int m, int n, unsigned int seed);
double vector_norm(double *vec, int size);
void gradient_check(double **X, double *y, double *theta, int m, int n, double epsilon, double tolerance);
void print_parameters(int m, int n, double epsilon, double tolerance, double *theta, int theta_size);
void get_user_input(int *m, int *n, double *epsilon, double *tolerance, double **theta, unsigned int *seed_choice);

// Entry point
int main() {
    int m, n;
    double epsilon, tolerance;
    double *theta = NULL;
    unsigned int seed_choice = 0;

    printf("=== Gradient Checking for Logistic Regression ===\n\n");

    // Get user input
    get_user_input(&m, &n, &epsilon, &tolerance, &theta, &seed_choice);

    // Allocate memory for dataset
    double **X = (double **)malloc(m * sizeof(double *));
    if (!X) {
        fprintf(stderr, "Memory allocation failed for X pointers\n");
        return 1;
    }
    for (int i = 0; i < m; i++) {
        X[i] = (double *)malloc(n * sizeof(double));
        if (!X[i]) {
            fprintf(stderr, "Memory allocation failed for X[%d]\n", i);
            // cleanup
            for (int k = 0; k < i; k++) free(X[k]);
            free(X);
            free(theta);
            return 1;
        }
    }
    double *y = (double *)malloc(m * sizeof(double));
    if (!y) {
        fprintf(stderr, "Memory allocation failed for y\n");
        for (int i = 0; i < m; i++) free(X[i]);
        free(X);
        free(theta);
        return 1;
    }

    // Generate synthetic dataset (optionally reproducible)
    printf("\nGenerating synthetic dataset with %d examples and %d features...\n", m, n);
    generate_dataset(X, y, m, n, seed_choice);

    // Print parameters
    print_parameters(m, n, epsilon, tolerance, theta, n);

    printf("Initial cost: %.6f\n", cost_function(X, y, theta, m, n));

    printf("\nPerforming gradient check...\n");
    // Numerical gradient needs 2 cost evaluations per parameter
    printf("This will require %d cost function evaluations (2 per parameter).\n", 2 * n);

    // Perform gradient checking
    gradient_check(X, y, theta, m, n, epsilon, tolerance);

    // Free memory
    for (int i = 0; i < m; i++) {
        free(X[i]);
    }
    free(X);
    free(y);
    free(theta);

    return 0;
}

// Get user input for parameters
void get_user_input(int *m, int *n, double *epsilon, double *tolerance, double **theta, unsigned int *seed_choice) {
    printf("Enter number of training examples (m) [default: %d]: ", DEFAULT_M);
    if (scanf("%d", m) != 1 || *m <= 0) {
        *m = DEFAULT_M;
    }

    printf("Enter number of features including bias (n) [default: %d]: ", DEFAULT_N);
    if (scanf("%d", n) != 1 || *n <= 1) {
        *n = DEFAULT_N;
    }

    printf("Enter epsilon (step size) [default: %.0e]: ", DEFAULT_EPSILON);
    if (scanf("%lf", epsilon) != 1 || *epsilon <= 0) {
        *epsilon = DEFAULT_EPSILON;
    }

    printf("Enter tolerance [default: %.0e]: ", DEFAULT_TOLERANCE);
    if (scanf("%lf", tolerance) != 1 || *tolerance <= 0) {
        *tolerance = DEFAULT_TOLERANCE;
    }

    // Consume leftover newline before asking for theta values
    int c;
    while ((c = getchar()) != '\n' && c != EOF) {}

    // Allocate memory for theta and get initial values
    *theta = (double *)malloc(*n * sizeof(double));
    if (!(*theta)) {
        fprintf(stderr, "Memory allocation failed for theta. Exiting.\n");
        exit(1);
    }

    printf("\nEnter initial parameter values for theta (%d values). Press Enter to use default 0.0 for a value.\n", *n);
    printf("Note: theta[0] is typically the bias term\n");

    // We'll read a line at a time to tolerate empty input
    for (int i = 0; i < *n; i++) {
        char line[128];
        printf("theta[%d]: ", i);
        if (!fgets(line, sizeof(line), stdin)) {
            // input failure: fallback to 0.0
            (*theta)[i] = 0.0;
            continue;
        }
        // strip newline
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        // if line is empty, default to 0.0 (safe)
        if (strlen(line) == 0) {
            (*theta)[i] = 0.0;
        } else {
            char *endptr = NULL;
            double val = strtod(line, &endptr);
            if (endptr == line) {
                // conversion failed
                (*theta)[i] = 0.0;
            } else {
                (*theta)[i] = val;
            }
        }
    }

    // Ask whether to use deterministic seed (reproducible dataset) or time-based seed
    printf("\nUse deterministic seed for dataset generation? (y/N): ");
    char answer[8];
    if (fgets(answer, sizeof(answer), stdin)) {
        if (answer[0] == 'y' || answer[0] == 'Y') {
            *seed_choice = 42; // deterministic seed
            printf("Deterministic seed chosen (seed=%u).\n", *seed_choice);
        } else {
            // derive seed from time
            *seed_choice = (unsigned int)time(NULL);
            printf("Non-deterministic seed chosen (seed=%u).\n", *seed_choice);
        }
    } else {
        *seed_choice = (unsigned int)time(NULL);
    }
}

// Print parameters with explanations
void print_parameters(int m, int n, double epsilon, double tolerance, double *theta, int theta_size) {
    printf("\n=== PARAMETER EXPLANATION ===\n");
    printf("Epsilon (ε = %.1e): Step size for finite differences\n", epsilon);
    printf("  - Used to perturb parameters for numerical gradient calculation\n");
    printf("  - Too large: Poor approximation of true derivative\n");
    printf("  - Too small: Numerical precision issues\n");
    printf("  - Recommended: 10⁻⁴ to 10⁻⁶\n\n");

    printf("Tolerance (%.1e): Maximum allowed difference between gradients\n", tolerance);
    printf("  - Determines if gradient check passes or fails\n");
    printf("  - Typical values: 10⁻⁷ to 10⁻⁹\n");
    printf("  - Difference < tolerance: Implementation is correct ✓\n");
    printf("  - Difference > tolerance: Potential bug in gradient code ✗\n\n");

    printf("=== CURRENT PARAMETERS ===\n");
    printf("Training examples (m): %d\n", m);
    printf("Features including bias (n): %d\n", n);
    printf("Epsilon (ε): %.1e\n", epsilon);
    printf("Tolerance: %.1e\n", tolerance);
    printf("Initial theta: [");
    for (int i = 0; i < theta_size; i++) {
        printf("%.6f%s", theta[i], (i < theta_size-1) ? ", " : "]\n");
    }
}

// Sigmoid function
double sigmoid(double z) {
    // clamp z to avoid overflow for very large negative/positive values
    if (z < -709.0) return 0.0; // exp(-709) ~ 1e-308
    if (z > 709.0) return 1.0;
    return 1.0 / (1.0 + exp(-z));
}

// Hypothesis for a single example (theta dot x)
double hypothesis(double *theta, double *x, int n_features) {
    double z = 0.0;
    for (int i = 0; i < n_features; i++) {
        z += theta[i] * x[i];
    }
    return sigmoid(z);
}

// Cost (average negative log-likelihood)
double cost_function(double **X, double *y, double *theta, int m, int n) {
    double total_cost = 0.0;

    for (int i = 0; i < m; i++) {
        double h = hypothesis(theta, X[i], n);
        // Use small epsilon inside log to avoid log(0)
        double clipped_h = fmax(1e-12, fmin(1.0 - 1e-12, h));
        if (y[i] == 1.0) {
            total_cost += -log(clipped_h);
        } else {
            total_cost += -log(1.0 - clipped_h);
        }
    }

    return total_cost / m;
}

// Analytical gradient (vectorized)
void analytical_gradient(double **X, double *y, double *theta, double *grad, int m, int n) {
    for (int j = 0; j < n; j++) {
        grad[j] = 0.0;
    }

    for (int i = 0; i < m; i++) {
        double h = hypothesis(theta, X[i], n);
        double error = h - y[i];
        for (int j = 0; j < n; j++) {
            grad[j] += error * X[i][j];
        }
    }

    for (int j = 0; j < n; j++) {
        grad[j] /= m;
    }
}

// Numerical gradient computed by in-place perturbation of theta (save/restore)
void numerical_gradient(double **X, double *y, double *theta, double *num_grad, int m, int n, double epsilon) {
    for (int j = 0; j < n; j++) {
        double orig = theta[j];

        theta[j] = orig + epsilon;
        double cost_plus = cost_function(X, y, theta, m, n);

        theta[j] = orig - epsilon;
        double cost_minus = cost_function(X, y, theta, m, n);

        // Restore original value
        theta[j] = orig;

        num_grad[j] = (cost_plus - cost_minus) / (2.0 * epsilon);
    }
}

// Generate synthetic dataset using randomly sampled true theta.
// seed == 42 means deterministic; otherwise seed from time or user choice.
void generate_dataset(double **X, double *y, int m, int n, unsigned int seed) {
    srand(seed);

    // True parameters for generating data
    double *true_theta = (double *)malloc(n * sizeof(double));
    if (!true_theta) {
        fprintf(stderr, "Memory allocation failed for true_theta\n");
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        // generate in [-1, 1]
        true_theta[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    for (int i = 0; i < m; i++) {
        // Bias term (x0 = 1)
        X[i][0] = 1.0;

        // Generate random features in [-1,1]
        for (int j = 1; j < n; j++) {
            X[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }

        // Compute probability using true parameters
        double z = 0.0;
        for (int j = 0; j < n; j++) {
            z += true_theta[j] * X[i][j];
        }
        double probability = sigmoid(z);

        // Generate label based on probability
        y[i] = (((double)rand() / RAND_MAX) < probability) ? 1.0 : 0.0;
    }

    free(true_theta);
}

// Euclidean norm of a vector
double vector_norm(double *vec, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

// Gradient check: compare analytical vs numerical
void gradient_check(double **X, double *y, double *theta, int m, int n, double epsilon, double tolerance) {
    double *anal_grad = (double *)malloc(n * sizeof(double));
    double *num_grad = (double *)malloc(n * sizeof(double));
    if (!anal_grad || !num_grad) {
        fprintf(stderr, "Memory allocation failed in gradient_check\n");
        exit(1);
    }

    // Compute both gradients
    analytical_gradient(X, y, theta, anal_grad, m, n);
    numerical_gradient(X, y, theta, num_grad, m, n, epsilon);

    printf("\n=== GRADIENT COMPARISON RESULTS ===\n");
    printf("Parameter | Analytical   | Numerical    | Relative Diff\n");
    printf("----------|--------------|--------------|----------------\n");

    for (int j = 0; j < n; j++) {
        double diff = fabs(anal_grad[j] - num_grad[j]);
        double avg = (fabs(anal_grad[j]) + fabs(num_grad[j])) / 2.0;
        double rel_diff = (avg > 1e-12) ? diff / avg : diff;
        printf("theta[%2d]  | %12.6e | %12.6e | %12.2e\n",
               j, anal_grad[j], num_grad[j], rel_diff);
    }

    // Compute overall difference metric
    double *diff_vector = (double *)malloc(n * sizeof(double));
    if (!diff_vector) {
        fprintf(stderr, "Memory allocation failed for diff_vector\n");
        free(anal_grad);
        free(num_grad);
        exit(1);
    }

    for (int j = 0; j < n; j++) {
        diff_vector[j] = anal_grad[j] - num_grad[j];
    }

    double norm_diff = vector_norm(diff_vector, n);
    double norm_anal = vector_norm(anal_grad, n);
    double norm_num = vector_norm(num_grad, n);
    double overall_diff = norm_diff / (norm_anal + norm_num + 1e-12);

    printf("\nOverall Difference Metric: %.2e\n", overall_diff);
    printf("Tolerance Threshold: %.0e\n", tolerance);

    if (overall_diff < tolerance) {
        printf("✓ GRADIENT CHECK PASSED! (Difference < tolerance)\n");
        printf("✓ Your analytical gradient implementation appears correct!\n");
    } else {
        printf("✗ GRADIENT CHECK FAILED! (Difference > tolerance)\n");
        printf("✗ There may be a bug in your analytical gradient code.\n");
        printf("  Suggestions:\n");
        printf("   - Verify the hypothesis and cost function implementations.\n");
        printf("   - Check bias term handling (theta[0] and X[:,0] == 1).\n");
        printf("   - Test with tiny datasets where you can compute gradients by hand.\n");
    }

    free(anal_grad);
    free(num_grad);
    free(diff_vector);
}
