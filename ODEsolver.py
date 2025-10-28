# ODEsolver.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation

class ODEsolver:
    def __init__(self, order, diffeqf, x, epochs, architecture, initializer, activation, optimizer):
        self.order = order
        self.diffeqf = diffeqf
        self.x = x
        self.epochs = epochs
        self.architecture = architecture
        self.activation = activation
        self.initializer = initializer
        self.optimizer = optimizer

        # Store epoch-by-epoch predictions
        self.epoch_predictions = []

        # Build NN model
        self.model = self.build_model()

    def build_model(self):
        inputs = tf.keras.Input(shape=(1,))
        x = inputs
        for units in self.architecture:
            x = tf.keras.layers.Dense(units, activation=self.activation, kernel_initializer=self.initializer)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.optimizer, loss='mse')
        return model

    def train(self, batch_size=32, print_every=1):
        history = {'loss': []}
        for epoch in range(1, self.epochs + 1):
            # Generate training data (here simple example: x vs exact solution)
            y_exact = self.exact(self.x)
            self.model.fit(self.x.reshape(-1,1), y_exact.reshape(-1,1), batch_size=batch_size, epochs=1, verbose=0)
            
            # Predict for animation
            y_pred_epoch = self.predict(self.x).flatten()
            self.epoch_predictions.append(y_pred_epoch)
            
            # Compute loss
            loss = np.mean((y_pred_epoch - y_exact) ** 2)
            history['loss'].append(loss)
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        return history

    def predict(self, x):
        return self.model.predict(x.reshape(-1,1), verbose=0)

    def exact(self, x):
        # Example: you can modify according to your ODE
        return (1.0 - x) * np.exp(x)

    def plot_solution(self, x_exact, y_exact, x_pred, y_pred, save_path=None):
        plt.figure()
        plt.plot(x_exact, y_exact, 'r', label='Exact')
        plt.scatter(x_pred, y_pred, color='b', label='NN Prediction')
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            print(f"Solution plot saved to {save_path}")
        plt.show()

    def plot_loss_curve(self, history, save_path=None):
        plt.figure()
        plt.semilogy(history['loss'], 'b')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            print(f"Loss curve saved to {save_path}")
        plt.show()

    def compute_error_metrics(self, y_exact, y_pred):
        mse = np.mean((y_exact - y_pred) ** 2)
        rel_l2 = np.linalg.norm(y_exact - y_pred) / np.linalg.norm(y_exact)
        return {'MSE': mse, 'Relative L2': rel_l2}

    def save_results(self, x, y, filename="results.csv"):
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for xi, yi in zip(x, y):
                writer.writerow([xi, yi])
        print(f"Results saved to {filename}")

    def get_epoch_predictions(self):
        return self.epoch_predictions

    # -------------------------------------------------
    # Training animation (Solution vs Exact + Error + Loss)
    # -------------------------------------------------
    def training_animation(self, x, y_exact, predictions, epochs, losses, save_path=None, step=100):
        """
        Animate training progress: NN predictions vs exact solution, error, and loss curve.

        Parameters:
            x           : array, spatial points
            y_exact     : array, exact solution at x
            predictions : list of predicted y arrays over epochs
            epochs      : array-like, epoch numbers
            losses      : array-like, training losses
            save_path   : str, path to save animation (optional)
            step        : int, skip every `step` epochs to reduce frames
        """
        import matplotlib.animation as animation

        fig, axes = plt.subplots(3, 1, figsize=(7, 10))
        ax1, ax2, ax3 = axes

        # -----------------------------
        # Top: Prediction vs Exact
        # -----------------------------
        ax1.plot(x, y_exact, "C1", label="Exact solution")
        ax1.set_ylabel("f(x)", fontsize=12)
        ax1.set_title("Neural Network Prediction vs Exact Solution")
        ax1.legend()
        ax1.set_xlim(min(x), max(x))
        ax1.set_ylim(min(y_exact)-0.1, max(y_exact)+0.1)

        # -----------------------------
        # Middle: Absolute error
        # -----------------------------
        ax2.set_ylabel("|Error|", fontsize=12)
        ax2.set_title("Absolute Error")
        ax2.set_xlim(min(x), max(x))
        ax2.set_ylim(0, max([np.max(np.abs(y_exact - y)) for y in predictions])*1.2)

        # -----------------------------
        # Bottom: Loss evolution
        # -----------------------------
        ax3.set_xlabel("Epochs", fontsize=12)
        ax3.set_ylabel("Loss", fontsize=12)
        ax3.set_title("Training Loss")
        ax3.set_xlim(min(epochs), max(epochs))
        ax3.set_ylim(min(losses)*0.9, max(losses)*1.1)
        ax3.semilogy(epochs, losses, color="lightgray", linewidth=1.0)

        frames = []
        x_loss_points, y_loss_points = [], []

        # Only keep every `step` frame
        for i in range(0, len(predictions), step):
            y_pred = predictions[i]
            # Prediction line
            line1, = ax1.plot(x, y_pred, ".", color="C0", markersize=3)
            # Error line
            line2, = ax2.plot(x, np.abs(y_exact - y_pred), ".", color="C2", markersize=3)
            # Loss point
            x_loss_points.append(epochs[i])
            y_loss_points.append(losses[i])
            line3, = ax3.semilogy(x_loss_points, y_loss_points, color="C3", linewidth=1.5)
            frames.append([line1, line2, line3])

        ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
        plt.tight_layout()

        if save_path is not None:
            ani.save(save_path, writer="pillow", fps=5)
            print(f"Animation saved to {save_path}")

        plt.show()
        return ani

