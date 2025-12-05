import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt


# PART 1b

def part1b():
    alpha = 1.0
    final_time = 15
    time_steps = [0.5, 1, 3, 5]
    x_nodes = np.arange(1, 52)
    N = len(x_nodes)

    T_init = np.zeros_like(x_nodes)
    T_init[0] = 1
    T_init[-1] = 1

    def exact_solution(x, t):
        return 0.5 * (1 + erf(x / (2 * sqrt(alpha * t))))

    for dt in time_steps:
        T = T_init.copy()
        A = np.zeros((N, N))
        b = T.copy()

        # BC
        A[0, 0], A[-1, -1] = 1, 1
        b[0], b[-1] = 1, 1

        for j in range(1, N - 1):
            dx_w = x_nodes[j] - x_nodes[j - 1]
            dx_e = x_nodes[j + 1] - x_nodes[j]
            dx_c = 0.5 * (x_nodes[j + 1] - x_nodes[j - 1])

            A[j, j - 1] = -alpha * dt / (dx_c * dx_w)
            A[j, j + 1] = -alpha * dt / (dx_c * dx_e)
            A[j, j] = 1 + alpha * dt * (1 / (dx_c * dx_w) + 1 / (dx_c * dx_e))

        T = np.linalg.solve(A, b)
        plt.plot(x_nodes, T, label=f"dt={dt}")

    T_exact = [exact_solution(xi, final_time) for xi in x_nodes]
    plt.plot(x_nodes, T_exact, 'k--', label='Exact (erf)')
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title('Backward Euler — Uniform Grid (t = 15)')
    plt.legend()
    plt.grid(True)

    filename = '/root/hw3/part1b_implicit_euler_uniform.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved Part 1B plot to {filename}")


# PART 1c

def part1c():
    alpha = 1.0
    final_time = 15
    time_steps = [0.5, 1, 3, 5]
    i_nodes = np.arange(1, 26)
    x_nodes = np.where(
        i_nodes <= 13,
        -0.6014 * abs(i_nodes - 13)**1.5,
        0.6014 * abs(i_nodes - 13)**1.5
    )
    N = len(x_nodes)

    T_init = np.zeros_like(x_nodes)
    T_init[:] = 1 

    def exact_solution(x, t):
        return 0.5 * (1 + erf(x / (2 * sqrt(alpha * t))))

    for dt in time_steps:
        T = T_init.copy()
        A = np.zeros((N, N))
        b = T.copy()

        A[0, 0], A[-1, -1] = 1, 1
        b[0], b[-1] = 1, 1

        for j in range(1, N - 1):
            dx_w = x_nodes[j] - x_nodes[j - 1]
            dx_e = x_nodes[j + 1] - x_nodes[j]
            dx_c = 0.5 * (x_nodes[j + 1] - x_nodes[j - 1])

            A[j, j - 1] = -alpha * dt / (dx_c * dx_w)
            A[j, j + 1] = -alpha * dt / (dx_c * dx_e)
            A[j, j] = 1 + alpha * dt * (1 / (dx_c * dx_w) + 1 / (dx_c * dx_e))

        T = np.linalg.solve(A, b)
        plt.plot(x_nodes, T, label=f"dt={dt}")

    # Analytical solution
    T_exact = [exact_solution(xi, final_time) for xi in x_nodes]
    plt.plot(x_nodes, T_exact, 'k--', label='Exact (erf)')
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title('Backward Euler — Nonuniform Grid (t = 15)')
    plt.legend()
    plt.grid(True)

    filename = '/root/hw3/part1c_implicit_euler_nonuniform.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved Part 1C plot to {filename}")


# PART 2c

def part2c():
    N = 21
    dx = 20 / (N - 1)
    dt = 0.01
    alpha = 1.0
    r = alpha * dt / (dx ** 2)

    A = np.zeros((N * N, N * N))
    for j in range(N):
        for i in range(N):
            idx = j * N + i
            if i == 0 or j == 0 or i == N - 1 or j == N - 1:
                A[idx, idx] = 1
            else:
                A[idx, idx] = 1 + 4 * r
                A[idx, idx - 1] = -r
                A[idx, idx + 1] = -r
                A[idx, idx - N] = -r
                A[idx, idx + N] = -r

    T = np.zeros((N, N))
    T[N // 2, N // 2] = 1

    num_steps = int(1 / dt)
    output_times = [0.25, 0.5, 1.0]
    output_steps = [int(t / dt) for t in output_times]

    for step in range(1, num_steps + 1):
        T_flat = T.flatten()
        T_new = np.linalg.solve(A, T_flat)
        T = T_new.reshape((N, N))

        if step in output_steps:
            current_time = round(step * dt, 2)
            plt.imshow(T, cmap='hot', origin='lower')
            plt.colorbar()
            plt.title(f"2D Diffusion – Temperature Field (t={current_time})")
            plt.xlabel('x')
            plt.ylabel('y')
            filename = f'/root/hw3/part2c_temperature_t{current_time}.png'
            plt.savefig(filename)
            plt.close()
            print(f"Saved Part 2C plot at t={current_time} to {filename}")



if __name__ == "__main__":
    part1b()
    part1c()
    part2c()
