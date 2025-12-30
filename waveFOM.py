import abc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import IPython.display
from matplotlib import cm
from scipy.linalg import expm

class waveFOM(abc.ABC):

    def __init__(self, N=6, f=1, Nt=100):
        self.N = N
        self.L = 2 * np.pi
        self.Nf = N*f
        self.Nt = Nt + 1
        self.x_fine = np.linspace(0, self.L, self.Nf, endpoint=False)
        self.x = np.linspace(0, self.L, self.N, endpoint=False)   
        self.Dmat = self.Dmat() 

    def IC(self, k=1, seed=0):
        rng = np.random.default_rng(seed)
        Nf = int(self.Nf)
        if Nf % 2 != 0:
            raise ValueError("Nf must be even")
        m = np.arange(0, Nf//2+1)                 # harmonics 0..K
        decay = 1.0 / (1.0 + m*m)               
        X = (self.x % (2*np.pi)).reshape(-1, 1) 

        out = []
        for _ in range(k):
            amp = 1.0 * rng.random(m.size) * decay
            phase = 2*np.pi * rng.random(m.size)
            u = np.sum(amp * np.cos(X*m - phase), axis=1)  
            out.append(u)

        return np.stack(out, axis=0)
    
    def Dmat(self):
        k = np.arange(self.N)
        j = k[:, None]
        K, J = np.meshgrid(k, k, indexing='ij')
        diff = K - J

        # Compute the matrix using the cotangent formula
        D = np.zeros((self.N, self.N), dtype=float)
        mask = K != J
        D[mask] = 0.5 * (-1) ** (K[mask] + J[mask]) / np.tan(np.pi * (K[mask] - J[mask]) / self.N)
        return D
    
    def rhs(self, t, u, c, gamma_q, gamma_p):
        if callable(c):
            c = c(t) 
        if callable(gamma_q):
            gamma_q = gamma_q(t) 
        if callable(gamma_p):
            gamma_p = gamma_p(t) 

        mid = len(u) // 2
        q, p = u[:mid], u[mid:]
        Aq = p - gamma_q*q
        Ap = c**2*self.Dmat@self.Dmat@q - gamma_p*p
        return np.hstack((Aq, Ap))
    
    def solve(self, Tf, u0, c, gamma_q, gamma_p, atol):
        t_span = (0, Tf)
        # Select appropriate RHS function
        rhs_func = (lambda t, u: self.rhs(t, u, c, gamma_q, gamma_p)) 

        # Call solver with or without t_eval
        sol = solve_ivp(rhs_func, t_span, u0, method='RK45', rtol=atol, atol=atol, dense_output=True)

        return sol.sol
    
    def create_grand_data_tensor(self, k, Tf, c, gamma_q, gamma_p, atol=1e-12, test_size=0.3):
        # Step 1: Generate k initial conditions
        initial_conditions = self.IC(k)  # shape: (k, N)
        initial_conditions2 = self.IC(k, seed=1)

        # Step 2: Solve the IVP for each IC and collect solutions
        solutions = []
        #t_eval = []
        for i in range(k):
            q0 = initial_conditions[i]
            p0 = initial_conditions2[i]
            u0 = np.hstack((q0,p0)).ravel()
            sol = self.solve(Tf, u0, c, gamma_q, gamma_p, atol)
            #t_eval.append(sol.t)  # shape: (T,)
            solutions.append(sol)  # shape: (T, N)
        # Step 3: Stack into shape (k, T, N)
        ind = int(round(k * test_size))
        return solutions[ind:], solutions[:ind]
    
    def animate(self, data, t_eval, k, skip=5):

        if data.ndim != 3:
            raise ValueError("Expected 3D data with shape (num_curves, N, Nt)")

        num_curves, N, Nt = data.shape
        if k > num_curves:
            raise ValueError(f"Requested k={k}, but data only has {num_curves} curves.")

        x = np.arange(N)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

        # Generate k distinct colors using a continuous colormap
        cmap = cm.get_cmap('tab20')  # Choose 'tab20', 'viridis', etc.
        colors = [cmap(i / k) for i in range(k)]

        # Create one line for each curve
        lines = []
        for i in range(k):
            line, = ax.plot([], [], '.-', color=colors[i], label=f'IC {i + 1}')
            lines.append(line)

        ax.set_xlim(0, N - 1)
        ax.set_ylim(np.min(data[:k]) * 1.1, np.max(data[:k]) * 1.1)
        ax.set_xlabel("x")
        ax.set_ylabel("Value")
        ax.set_title(r"$t = 0$")
        ax.legend(loc='upper right')

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(frame):
            for i, line in enumerate(lines):
                line.set_data(x, data[i, :, frame * skip])
            ax.set_title(rf"$t = {round(t_eval[frame * skip],3)}$")
            return lines

        ani = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=Nt // skip,
            interval=100,
            blit=True
        )

        plt.close(fig)
        return IPython.display.HTML(ani.to_jshtml())
    
    