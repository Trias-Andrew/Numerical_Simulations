import numpy as np
import matplotlib.pyplot as plt

def lotka_volterra(t, y, alpha=1.1, beta=0.4, delta=0.1, gamma=0.4):
    """
    Lotka-Volterra equations (predator-prey model).
    y[0]: prey population
    y[1]: predator population
    """
    prey, predator = y
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    return np.array([dprey_dt, dpredator_dt])

def rk4_step(f, t, y, dt, **kwargs):
    """
    Perform one step of the RK4 method.
    """
    k1 = f(t, y, **kwargs)
    k2 = f(t + dt/2, y + dt/2 * k1, **kwargs)
    k3 = f(t + dt/2, y + dt/2 * k2, **kwargs)
    k4 = f(t + dt, y + dt * k3, **kwargs)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

def simulate(f, y0, t0, t_end, dt, **kwargs):
    """
    Simulate the system from t0 to t_end with step size dt.
    """
    times = np.arange(t0, t_end + dt, dt)
    y = np.zeros((len(times), len(y0)))
    y[0] = y0

    for i in range(1, len(times)):
        y[i] = rk4_step(f, times[i-1], y[i-1], dt, **kwargs)
    
    return times, y

def plot_results(times, y, labels):
    """
    Plot the time evolution of the system variables.
    """
    for i in range(y.shape[1]):
        plt.plot(times, y[:, i], label=labels[i])
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Lotka-Volterra Predator-Prey Dynamics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Parameters
y0 = [10, 5]  # Initial populations: [prey, predator]
t0 = 0
t_end = 50
dt = 0.01

# Run simulation
times, y = simulate(lotka_volterra, y0, t0, t_end, dt)

# Plot results
plot_results(times, y, labels=['Prey', 'Predator'])
