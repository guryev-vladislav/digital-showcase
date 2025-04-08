import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def linear_regression_analysis(a_true, b_true, sigma_sq, n, m):
    """
    Performs one-dimensional linear regression analysis.

    Args:
        a_true (float): The true slope coefficient.
        b_true (float): The true intercept coefficient.
        sigma_sq (float): The variance of the random errors.
        n (int): The size of the first sample.
        m (int): The size of the additional sample.

    Returns:
        tuple: A tuple containing:
            - a_estimated (float): The estimated slope coefficient.
            - b_estimated (float): The estimated intercept coefficient.
            - r_squared (float): The coefficient of determination R^2.
            - y_predicted_additional (np.ndarray): The predicted values for the additional sample.
            - y_additional (np.ndarray): The true values for the additional sample.
    """
    # Step 1: Input coefficients and get the first sample
    x = np.arange(1, n + 1)
    epsilon = np.random.normal(0, np.sqrt(sigma_sq), n)
    y = a_true * x + b_true + epsilon

    # Step 2: Estimate the linear regression coefficients
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    a_estimated = slope
    b_estimated = intercept
    r_squared = r_value**2

    # Step 3: Calculate the coefficient of determination R^2
    print(f"Estimated slope coefficient (a*): {a_estimated:.4f}")
    print(f"Estimated intercept coefficient (b*): {b_estimated:.4f}")
    print(f"Coefficient of determination (R^2): {r_squared:.4f}")

    # Step 4: Get an additional sample and compare predicted values
    x_additional = np.arange(n + 1, n + m + 1)
    epsilon_additional = np.random.normal(0, np.sqrt(sigma_sq), m)
    y_additional = a_true * x_additional + b_true + epsilon_additional
    y_predicted_additional = a_estimated * x_additional + b_estimated

    print("\nComparison of predicted and true values for the additional sample:")
    for i in range(m):
        print(f"x = {x_additional[i]}, y_true = {y_additional[i]:.4f}, y_predicted = {y_predicted_additional[i]:.4f}")

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='First sample (training)')
    plt.plot(x, a_true * x + b_true, 'g-', label=f'True line: y = {a_true}x + {b_true}')
    plt.plot(x, a_estimated * x + b_estimated, 'r--', label=f'Estimated line: ŷ = {a_estimated:.2f}x + {b_estimated:.2f}')
    plt.scatter(x_additional, y_additional, label='Additional sample (testing)')
    plt.plot(x_additional, y_predicted_additional, 'm--', label='Predicted values for additional sample')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('One-Dimensional Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

    return a_estimated, b_estimated, r_squared, y_predicted_additional, y_additional

if __name__ == "__main__":
    # Define true parameters and sample sizes
    true_a = 2.5
    true_b = 1.0
    error_sigma_sq = 4.0
    n_samples = 50
    m_additional_samples = 20

    # Run the linear regression analysis
    estimated_a, estimated_b, r2, y_pred_add, y_add = linear_regression_analysis(true_a, true_b, error_sigma_sq, n_samples, m_additional_samples)