#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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
void generate_dataset(double **X, double *y, int m, int n);
double vector_norm(double *vec, int size);
void gradient_check(double **X, double *y, double *theta, int m, int n, double epsilon, double tolerance);
void print_parameters(int m, int n, double epsilon, double tolerance, double *theta, int theta_size);
void get_user_input(int *m, int *n, double *epsilon, double *tolerance, double **theta);

int main() {
    int m, n;
    double epsilon, tolerance;
    double *theta = NULL;

    printf("=== Gradient Checking for Logistic Regression ===\n\n");

    // Get user input
    get_user_input(&m, &n, &epsilon, &tolerance, &theta);

    // Allocate memory for dataset
    double **X = (double **)malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++) {
        X[i] = (double *)malloc(n * sizeof(double));
    }
    double *y = (double *)malloc(m * sizeof(double));

    // Generate synthetic dataset
    printf("\nGenerating synthetic dataset with %d examples and %d features...\n", m, n);
    generate_dataset(X, y, m, n);

    // Print parameters
    print_parameters(m, n, epsilon, tolerance, theta, n);

    printf("Initial cost: %.6f\n", cost_function(X, y, theta, m, n));

    printf("\nPerforming gradient check...\n");
    printf("This will require %d cost function evaluations...\n", 2 * n * m);

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
void get_user_input(int *m, int *n, double *epsilon, double *tolerance, double **theta) {
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

    // Allocate memory for theta and get initial values
    *theta = (double *)malloc(*n * sizeof(double));
    printf("\nEnter initial parameter values for theta (%d values):\n", *n);
    printf("Note: theta[0] is typically the bias term\n");

    for (int i = 0; i < *n; i++) {
        printf("theta[%d]: ", i);
        if (scanf("%lf", &(*theta)[i]) != 1) {
            // Set default values if input fails
            double default_theta[] = {0.1, -0.2, 0.05, 0.3, -0.15};
            (*theta)[i] = (i < *n) ? default_theta[i] : 0.0;
        }
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
        printf("%.3f%s", theta[i], (i < theta_size-1) ? ", " : "]\n");
    }
}

// [Rest of the functions remain the same as previous implementation]
// sigmoid(), hypothesis(), cost_function(), analytical_gradient(),
// numerical_gradient(), generate_dataset(), vector_norm(), gradient_check()

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double hypothesis(double *theta, double *x, int n_features) {
    double z = 0.0;
    for (int i = 0; i < n_features; i++) {
        z += theta[i] * x[i];
    }
    return sigmoid(z);
}


double cost_function(double **X, double *y, double *theta, int m, int n) {
    double total_cost = 0.0;

    for (int i = 0; i < m; i++) {
        double h = hypothesis(theta, X[i], n);
        if (y[i] == 1) {
            total_cost += -log(h + 1e-10); // Avoid log(0)
        } else {
            total_cost += -log(1.0 - h + 1e-10);
        }
    }

    return total_cost / m;
}

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

void numerical_gradient(double **X, double *y, double *theta, double *num_grad, int m, int n, double epsilon) {
    double *theta_plus = (double *)malloc(n * sizeof(double));
    double *theta_minus = (double *)malloc(n * sizeof(double));

    for (int j = 0; j < n; j++) {
        // Create theta_plus and theta_minus
        for (int k = 0; k < n; k++) {
            theta_plus[k] = theta[k];
            theta_minus[k] = theta[k];
        }

        theta_plus[j] += epsilon;
        theta_minus[j] -= epsilon;

        double cost_plus = cost_function(X, y, theta_plus, m, n);
        double cost_minus = cost_function(X, y, theta_minus, m, n);

        num_grad[j] = (cost_plus - cost_minus) / (2 * epsilon);
    }

    free(theta_plus);
    free(theta_minus);
}

void generate_dataset(double **X, double *y, int m, int n) {
    srand(time(NULL));

    // True parameters for generating data
    double *true_theta = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        true_theta[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }

    for (int i = 0; i < m; i++) {
        // Bias term (x0 = 1)
        X[i][0] = 1.0;

        // Generate random features
        for (int j = 1; j < n; j++) {
            X[i][j] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        }

        // Compute probability using true parameters
        double z = 0.0;
        for (int j = 0; j < n; j++) {
            z += true_theta[j] * X[i][j];
        }
        double probability = sigmoid(z);

        // Generate label based on probability
        y[i] = ((double)rand() / RAND_MAX) < probability ? 1.0 : 0.0;
    }

    free(true_theta);
}

double vector_norm(double *vec, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

void gradient_check(double **X, double *y, double *theta, int m, int n, double epsilon, double tolerance) {
    double *anal_grad = (double *)malloc(n * sizeof(double));
    double *num_grad = (double *)malloc(n * sizeof(double));

    // Compute both gradients
    analytical_gradient(X, y, theta, anal_grad, m, n);
    numerical_gradient(X, y, theta, num_grad, m, n, epsilon);

    printf("\n=== GRADIENT COMPARISON RESULTS ===\n");
    printf("Parameter | Analytical  | Numerical   | Relative Diff\n");
    printf("---------|-------------|-------------|-------------\n");

    for (int j = 0; j < n; j++) {
        double diff = fabs(anal_grad[j] - num_grad[j]);
        double avg = (fabs(anal_grad[j]) + fabs(num_grad[j])) / 2.0;
        double rel_diff = (avg > 1e-10) ? diff / avg : diff;

        printf("theta[%d] | %-10.6f | %-10.6f | %-10.2e\n",
               j, anal_grad[j], num_grad[j], rel_diff);
    }

    // Compute overall difference metric
    double *diff_vector = (double *)malloc(n * sizeof(double));
    for (int j = 0; j < n; j++) {
        diff_vector[j] = anal_grad[j] - num_grad[j];
    }

    double norm_diff = vector_norm(diff_vector, n);
    double norm_anal = vector_norm(anal_grad, n);
    double norm_num = vector_norm(num_grad, n);
    double overall_diff = norm_diff / (norm_anal + norm_num + 1e-10);

    printf("\nOverall Difference Metric: %.2e\n", overall_diff);
    printf("Tolerance Threshold: %.0e\n", tolerance);

    if (overall_diff < tolerance) {
        printf("✓ GRADIENT CHECK PASSED! (Difference < tolerance)\n");
        printf("✓ Your analytical gradient implementation appears correct!\n");
    } else {
        printf("✗ GRADIENT CHECK FAILED! (Difference > tolerance)\n");
        printf("✗ There may be a bug in your analytical gradient code.\n");
        printf("   Check your derivative calculations and implementation.\n");
    }

    free(anal_grad);
    free(num_grad);
    free(diff_vector);
}
///
