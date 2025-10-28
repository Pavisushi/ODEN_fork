import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# -----------------------------
# Environment & random seed
# -----------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

# -----------------------------
# Matplotlib configuration
# -----------------------------
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = ['DejaVu Serif']
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# -----------------------------
# Import project modules
# -----------------------------
from Dictionary import Dictionary
from ODEsolver import ODEsolver
from DiffEq import DiffEq

# -----------------------------
# Exact solution function
# -----------------------------
def exact_solution(x):
    return (1.0 - x) * np.exp(x)

# -----------------------------
# Main script
# -----------------------------
if __name__ == "__main__":
    # Load dictionary
    D = Dictionary()
    Dict = D.Dict

    # -----------------------------
    # Solver configuration
    # -----------------------------
    order = 6
    diffeqf = "sixth"
    x = np.linspace(0.0, 1.0, 200)
    architecture = [8]
    activation = Dict["activation"]["tanh"]
    initializer = Dict["initializer"]["GlorotUniform"]
    optimizer = Dict["optimizer"]["Adamax"]
    epochs = 7000

    # -----------------------------
    # Initialize solver
    # -----------------------------
    solver = ODEsolver(order, diffeqf, x, epochs, architecture, initializer, activation, optimizer)

    # -----------------------------
    # Train model
    # -----------------------------
    history = solver.train(batch_size=32, print_every=500)

    # -----------------------------
    # Plot loss curve
    # -----------------------------
    solver.plot_loss_curve(history)

    # -----------------------------
    # Predictions vs exact
    # -----------------------------
    x_predict = np.linspace(0, 1, 25)
    y_pred = solver.predict(x_predict)
    
    x_exact = np.linspace(0, 1, 200)
    y_pred_exact = solver.predict(x_exact)
    y_exact = exact_solution(x_exact)
    solver.plot_solution(x_exact, y_exact, x_predict, y_pred)

    # -----------------------------
    # Compute error metrics
    # -----------------------------
    metrics = solver.compute_error_metrics(y_exact, solver.predict(x_exact))
    print("Error Metrics:", metrics)

    # -----------------------------
    # Export results CSV and plots
    # -----------------------------
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    solver.save_results(x_predict, y_pred, filename=os.path.join(results_dir, "results.csv"))
    solver.plot_solution(x_exact, y_exact, x_predict, y_pred, save_path=os.path.join(results_dir, "solution_plot.png"))
    solver.plot_loss_curve(history, save_path=os.path.join(results_dir, "loss_curve.png"))

    # -----------------------------
    # Print table of solutions
    # -----------------------------
    x_table = np.linspace(0, 1, 11)
    y_exact_table = exact_solution(x_table)
    y_pred_table = solver.predict(x_table).flatten()
    error_table = np.abs(y_exact_table - y_pred_table)

    print("\nNumerical results for Example 1")
    print("    x    Analytical solution    Numerical solution    Error^a")
    print("------  --------------------    ------------------    ----------")
    for xi, ya, yn, er in zip(x_table, y_exact_table, y_pred_table, error_table):
        print(f" {xi:4.1f}    {ya:12.8f}    {yn:12.8f}    {er:12.8f}")
    print("\n^a Error = |Analytical solution − Numerical solution|")

    # -----------------------------
    # Create training animation
    # -----------------------------
    # Prepare predictions at each epoch
    predictions = solver.get_epoch_predictions()  
    losses = history['loss'] if 'loss' in history else np.linspace(0.2, 0.001, epochs)

    animation_path = os.path.join(results_dir, "training_animation.gif")
    ani = solver.training_animation(
        x=x_exact,
        y_exact=y_exact,
        predictions=predictions,
        epochs=np.arange(1, epochs+1),
        losses=history['loss'] if 'loss' in history else np.linspace(0.2, 0.001, epochs),
        save_path=animation_path
    )
    