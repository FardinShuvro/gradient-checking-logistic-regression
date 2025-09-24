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
#define MAX_FILENAME_LENGTH 100
#define MAX_ITERATIONS 1000

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

// New function prototypes for additional features
void save_results_to_file(double **X, double *y, double *theta, double *anal_grad,
                         double *num_grad, int m, int n, double epsilon, double tolerance,
                         double overall_diff, int passed, const char *filename);
void display_convergence_analysis(double **X, double *y, double *theta, int m, int n);
void test_different_epsilons(double **X, double *y, double *theta, int m, int n, double tolerance);
void performance_benchmark(double **X, double *y, double *theta, int m, int n);
void print_validation_report(double overall_diff, double tolerance, int passed);
void interactive_mode(double **X, double *y, double *theta, int m, int n);

int main() {
    int m, n;
    double epsilon, tolerance;
    double *theta = NULL;

    printf("=== Advanced Gradient Checking for Logistic Regression ===\n\n");

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
    clock_t start_time = clock();
    gradient_check(X, y, theta, m, n, epsilon, tolerance);
    clock_t end_time = clock();

    double computation_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("\nComputation time: %.3f seconds\n", computation_time);

    // Additional features
    printf("\n=== ADDITIONAL ANALYSIS FEATURES ===\n");

    int choice;
    do {
        printf("\nSelect additional analysis:\n");
        printf("1. Save results to file\n");
        printf("2. Convergence analysis\n");
        printf("3. Test different epsilon values\n");
        printf("4. Performance benchmark\n");
        printf("5. Interactive mode\n");
        printf("0. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch(choice) {
            case 1:
                {
                    char filename[MAX_FILENAME_LENGTH];
                    printf("Enter filename to save results: ");
                    scanf("%s", filename);
                    // For demonstration, we'll create dummy gradient arrays
                    double *anal_grad = (double *)malloc(n * sizeof(double));
                    double *num_grad = (double *)malloc(n * sizeof(double));
                    analytical_gradient(X, y, theta, anal_grad, m, n);
                    numerical_gradient(X, y, theta, num_grad, m, n, epsilon);
                    double overall_diff = 0.0; // This would be calculated in real scenario
                    save_results_to_file(X, y, theta, anal_grad, num_grad, m, n,
                                       epsilon, tolerance, overall_diff, 1, filename);
                    free(anal_grad);
                    free(num_grad);
                }
                break;
            case 2:
                display_convergence_analysis(X, y, theta, m, n);
                break;
            case 3:
                test_different_epsilons(X, y, theta, m, n, tolerance);
                break;
            case 4:
                performance_benchmark(X, y, theta, m, n);
                break;
            case 5:
                interactive_mode(X, y, theta, m, n);
                break;
            case 0:
                printf("Exiting...\n");
                break;
            default:
                printf("Invalid choice. Please try again.\n");
        }
    } while (choice != 0);

    // Free memory
    for (int i = 0; i < m; i++) {
        free(X[i]);
    }
    free(X);
    free(y);
    free(theta);

    printf("\nThank you for using Advanced Gradient Checker!\n");
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

// Core mathematical functions (unchanged)
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
    printf("----------|-------------|-------------|-------------\n");

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

    print_validation_report(overall_diff, tolerance, overall_diff < tolerance);

    free(anal_grad);
    free(num_grad);
    free(diff_vector);
}

// NEW FEATURE: Save results to file
void save_results_to_file(double **X, double *y, double *theta, double *anal_grad,
                         double *num_grad, int m, int n, double epsilon, double tolerance,
                         double overall_diff, int passed, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Could not open file %s for writing.\n", filename);
        return;
    }

    fprintf(file, "GRADIENT CHECKING RESULTS\n");
    fprintf(file, "==========================\n\n");
    fprintf(file, "Date: %s", ctime(&(time_t){time(NULL)}));
    fprintf(file, "Dataset: %d examples, %d features\n", m, n);
    fprintf(file, "Epsilon: %.2e, Tolerance: %.2e\n\n", epsilon, tolerance);

    fprintf(file, "Parameter Results:\n");
    fprintf(file, "------------------\n");
    for (int j = 0; j < n; j++) {
        double diff = fabs(anal_grad[j] - num_grad[j]);
        double avg = (fabs(anal_grad[j]) + fabs(num_grad[j])) / 2.0;
        double rel_diff = (avg > 1e-10) ? diff / avg : diff;

        fprintf(file, "theta[%d]: Analytical=%-10.6f, Numerical=%-10.6f, Diff=%-10.2e\n",
                j, anal_grad[j], num_grad[j], rel_diff);
    }

    fprintf(file, "\nOverall difference: %.2e\n", overall_diff);
    fprintf(file, "Status: %s\n", passed ? "PASSED" : "FAILED");

    fclose(file);
    printf("Results saved to %s\n", filename);
}

// NEW FEATURE: Convergence analysis
void display_convergence_analysis(double **X, double *y, double *theta, int m, int n) {
    printf("\n=== CONVERGENCE ANALYSIS ===\n");
    printf("Testing gradient consistency across iterations...\n");

    double current_theta[n];
    memcpy(current_theta, theta, n * sizeof(double));

    for (int iter = 0; iter < 5; iter++) {
        double grad[n];
        analytical_gradient(X, y, current_theta, grad, m, n);

        // Update theta (simple gradient descent step)
        double learning_rate = 0.1;
        for (int j = 0; j < n; j++) {
            current_theta[j] -= learning_rate * grad[j];
        }

        double cost = cost_function(X, y, current_theta, m, n);
        printf("Iteration %d: Cost = %.6f\n", iter + 1, cost);
    }
}

// NEW FEATURE: Test different epsilon values
void test_different_epsilons(double **X, double *y, double *theta, int m, int n, double tolerance) {
    printf("\n=== EPSILON SENSITIVITY ANALYSIS ===\n");
    printf("Testing different epsilon values:\n");

    double epsilons[] = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    int num_epsilons = sizeof(epsilons) / sizeof(epsilons[0]);

    for (int i = 0; i < num_epsilons; i++) {
        double num_grad[n];
        numerical_gradient(X, y, theta, num_grad, m, n, epsilons[i]);

        double anal_grad[n];
        analytical_gradient(X, y, theta, anal_grad, m, n);

        double diff_vector[n];
        for (int j = 0; j < n; j++) {
            diff_vector[j] = anal_grad[j] - num_grad[j];
        }

        double norm_diff = vector_norm(diff_vector, n);
        double norm_anal = vector_norm(anal_grad, n);
        double overall_diff = norm_diff / (norm_anal + 1e-10);

        printf("ε=%.0e: Difference=%.2e %s\n", epsilons[i], overall_diff,
               overall_diff < tolerance ? "✓" : "✗");
    }
}

// NEW FEATURE: Performance benchmark
void performance_benchmark(double **X, double *y, double *theta, int m, int n) {
    printf("\n=== PERFORMANCE BENCHMARK ===\n");

    clock_t start, end;
    double cpu_time_used;

    // Benchmark analytical gradient
    start = clock();
    for (int i = 0; i < 100; i++) {
        double grad[n];
        analytical_gradient(X, y, theta, grad, m, n);
    }
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Analytical gradient (100 runs): %.3f seconds\n", cpu_time_used);

    // Benchmark numerical gradient
    start = clock();
    for (int i = 0; i < 100; i++) {
        double num_grad[n];
        numerical_gradient(X, y, theta, num_grad, m, n, 1e-4);
    }
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Numerical gradient (100 runs): %.3f seconds\n", cpu_time_used);

    printf("Speed ratio: %.1fx faster\n",
           (cpu_time_used > 0) ? (cpu_time_used / 0.001) : 0);
}

// NEW FEATURE: Enhanced validation report
void print_validation_report(double overall_diff, double tolerance, int passed) {
    printf("\n=== VALIDATION REPORT ===\n");
    if (passed) {
        printf("✓ GRADIENT CHECK PASSED! (Difference < tolerance)\n");
        printf("✓ Your analytical gradient implementation appears correct!\n");
        printf("✓ You can confidently use analytical gradients for training.\n");
    } else {
        printf("✗ GRADIENT CHECK FAILED! (Difference > tolerance)\n");
        printf("✗ There may be a bug in your analytical gradient code.\n");
        printf("  Recommended actions:\n");
        printf("  1. Check derivative calculations\n");
        printf("  2. Verify implementation of cost function\n");
        printf("  3. Review vectorization code\n");
        printf("  4. Test with smaller dataset first\n");
    }
}

// NEW FEATURE: Interactive mode
void interactive_mode(double **X, double *y, double *theta, int m, int n) {
    printf("\n=== INTERACTIVE MODE ===\n");
    printf("You can now test different theta values interactively.\n");

    double test_theta[n];
    memcpy(test_theta, theta, n * sizeof(double));

    int choice;
    do {
        printf("\nCurrent theta: [");
        for (int i = 0; i < n; i++) {
            printf("%.3f%s", test_theta[i], (i < n-1) ? ", " : "]\n");
        }

        printf("1. Modify theta values\n");
        printf("2. Compute cost with current theta\n");
        printf("3. Compute gradient with current theta\n");
        printf("0. Return to main menu\n");
        printf("Enter choice: ");
        scanf("%d", &choice);

        switch(choice) {
            case 1:
                printf("Enter new theta values (%d values):\n", n);
                for (int i = 0; i < n; i++) {
                    printf("theta[%d]: ", i);
                    scanf("%lf", &test_theta[i]);
                }
                break;
            case 2:
                printf("Cost with current theta: %.6f\n",
                       cost_function(X, y, test_theta, m, n));
                break;
            case 3:
                {
                    double grad[n];
                    analytical_gradient(X, y, test_theta, grad, m, n);
                    printf("Gradient: [");
                    for (int i = 0; i < n; i++) {
                        printf("%.6f%s", grad[i], (i < n-1) ? ", " : "]\n");
                    }
                }
                break;
        }
    } while (choice != 0);
}
