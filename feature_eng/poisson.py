import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    
    model = PoissonRegression(step_size=lr)
    model.fit(x_train, y_train)
    prediction = model.predict(x_eval)
    np.savetxt(save_path, prediction)
    
    plt.scatter(y_eval, prediction)
    plt.xlabel('True Count')
    plt.ylabel('Predicted Count')
    plt.title('True Counts vs Predicted Counts for Traffic Data')
    plt.show()


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-3, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
            10000000
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

        
    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Update the parameter by step_size * (sum of the gradient over examples)

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        n_rows = x.shape[0]
        n_cols = x.shape[1]
        self.theta = np.zeros(n_cols)
        num_iter = 0
        while num_iter <= self.max_iter:
            diff = y - np.exp(np.dot(x, self.theta))              
            gradient = np.dot(np.transpose(x), diff)
            gradient *= self.step_size
            if np.linalg.norm(gradient) >= self.eps:
                self.theta += (gradient)
            else: 
                break
            num_iter += 1              
        

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        return np.exp(x.dot(self.theta))
    
    

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
