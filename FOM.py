import abc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation, HTMLWriter
from scipy.integrate import solve_ivp
import IPython.display
from matplotlib import cm
from scipy.linalg import expm


class FOM(abc.ABC):

    def __init__(self, N=6, f=1, Nt=100, forcing=False):
        self.N = N
        self.L = 2 * np.pi
        self.Nf = int(N*f)
        self.Nt = Nt + 1
        self.x_fine = np.linspace(0, self.L, self.Nf, endpoint=False)
        self.x = np.linspace(0, self.L, self.N, endpoint=False)   
        self.Dmat = self.Dmat() 
        self.Dmat2 = self.Dmat@self.Dmat
    

    def IC(self, k=1, seed=0):
        rng = np.random.default_rng(seed)
        Nf = self.Nf
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
    
    def rhs(self, t, u, mu, a, v):
        mu = mu(t) if callable(mu) else mu
        v = v(t) if callable(v) else v
        a = a(t) if callable(a) else a
        n = self.Dmat.shape[0]
        rhs = np.diag([mu]*n)@self.Dmat2
        rhs += -np.diag([v]*n)@self.Dmat
        rhs += np.diag([a]*n)
        rhsu = rhs@u
        # if self.forcing: 
        #     rhsu += self.f(t)
        return rhsu
    
    
    def rhs2(self, u, mu, a, v):
        if mu.ndim != 2 and v.ndim !=2: 
            raise Exception("mu and v need to be diagonal matrices")
        rhs = mu@self.Dmat@self.Dmat
        rhs += - v@self.Dmat 
        rhs += np.diag([a]*self.Dmat.shape[0])
        return rhs@u
    
    def f(self, u0, forcing):
        if forcing == "rank1":
            return lambda t: 0*u0 + .001 #1/(1+t) 
        elif forcing == "full_rank":
            return lambda t: u0/(1 + t)
        else:
            raise Exception("Only options are: 'rank1', 'full_rank'")
        
    
    def solve(self, Tf, u0, mu, a, v, atol, forcing):
        t_span = (0, Tf)
        if forcing != None:
            f = self.f(u0, forcing)
        else:
            f = lambda t: 0
        # Select appropriate RHS function
        rhs_func = lambda t, u: self.rhs(t, u, mu, a, v) + f(t) 

        # Call solver with or without t_eval
        sol = solve_ivp(rhs_func, t_span, u0, method='RK45', rtol=atol, atol=atol, dense_output=True)

        return sol.sol
    
    def rhsMat(self, t, mu, a, v):
        mu = mu(t) if callable(mu) else mu
        v = v(t) if callable(v) else v
        a = a(t) if callable(a) else a
        n = self.Dmat.shape[0]
        rhs = np.diag([mu]*n)@self.Dmat2
        rhs += -np.diag([v]*n)@self.Dmat
        rhs += np.diag([a]*n)
        return rhs
    
    
    def R(self, t, mu, a, v, indices):
        A = self.rhsMat(t, mu, a, v)
        indices_resolved = indices
        R = A[np.ix_(indices_resolved, indices_resolved)]
        return R
    
    def RTilde(self, t, mu, a, v, indices):
        A = self.rhsMat(t, mu, a, v)
        indices_resolved = indices
        indices_unresolved = [i for i in range(len(self.x)) if i not in indices]
        R = A[np.ix_(indices_resolved, indices_unresolved)]
        return R


    def memory(self, t, mu, a, v, indices):
        if callable(mu) or callable(a) or callable(v):
            raise NotImplementedError
        A = self.rhsMat(0, mu, a, v)
        indices_resolved = indices
        indices_unresolved = [i for i in range(len(self.x)) if i not in indices]
        R = A[np.ix_(indices_resolved, indices_resolved)]  # Resolved -> Resolved
        RTilde = A[np.ix_(indices_resolved, indices_unresolved)]  # Resolved -> Unresolved
        U = A[np.ix_(indices_unresolved, indices_resolved)]  # Unresolved -> Resolved
        UTilde = A[np.ix_(indices_unresolved, indices_unresolved)]
        K = np.zeros((RTilde.shape[0], U.shape[1], len(t)))
        for n in range(len(t)):
            K[:,:,n] = RTilde@expm(t[n]*UTilde)@U
        return K


    def create_grand_data_tensor(self, k, Tf, mu0, a, v, forcing=None, atol=1e-12, test_size=0.3):
        # Step 1: Generate k initial conditions
        initial_conditions = self.IC(k)  # shape: (k, N)

        # Step 2: Solve the IVP for each IC and collect solutions
        solutions = []
        #t_eval = []
        for i in range(k):
            sol = self.solve(Tf, initial_conditions[i], mu0, a, v, forcing=forcing, atol=atol)
            #t_eval.append(sol.t)  # shape: (T,)
            solutions.append(sol)  # shape: (T, N)
        # Step 3: Stack into shape (k, T, N)
        ind = int(round(k * test_size))
        return solutions[ind:], solutions[:ind]


    def split_matrix_by_columns(self, full_matrix, v):
        """
        Returns:
            M1 (ndarray): Matrix with selected columns
            M2 (ndarray): Matrix with remaining columns
        """
        v = np.array(v)
        all_indices = np.arange(full_matrix.shape[1])
        rest_indices = np.setdiff1d(all_indices, v)

        M1 = full_matrix[:, v]
        M2 = full_matrix[:, rest_indices]

        return M1, M2

    def animate(self, data, t_eval, k, skip=5, interval=100, save_html_path=None, method="writer"):
        
        if data.ndim != 3:
            raise ValueError("Expected 3D data with shape (num_curves, N, Nt)")

        num_curves, N, Nt = data.shape
        if k > num_curves:
            raise ValueError(f"Requested k={k}, but data only has {num_curves} curves.")

        x = np.arange(N)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

        cmap = cm.get_cmap('tab20')
        colors = [cmap(i / max(1, k)) for i in range(k)]

        lines = []
        for i in range(k):
            line, = ax.plot([], [], '.-', color=colors[i], label=f'IC {i + 1}')
            lines.append(line)

        ax.set_xlim(0, N - 1)
        ylo = float(np.min(data[:k])) * 1.1
        yhi = float(np.max(data[:k])) * 1.1
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel("x")
        ax.set_ylabel("Value")
        ax.set_title(r"$t = 0$")
        ax.legend(loc='upper right')

        nframes = Nt // skip

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(frame):
            j = frame * skip
            for i, line in enumerate(lines):
                line.set_data(x, data[i, :, j])
            ax.set_title(rf"$t = {round(float(t_eval[j]), 3)}$")
            return lines

        ani = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=nframes,
            interval=interval,
            blit=True
        )

        if save_html_path:
            if method == "writer":
                fps = 1000.0 / float(interval)
                writer = HTMLWriter(fps=fps, embed_frames=True, default_mode="once")
                ani.save(save_html_path, writer=writer)
                plt.close(fig)
                return None
            elif method == "js":
                html = ani.to_jshtml()
                with open(save_html_path, "w", encoding="utf-8") as f:
                    f.write(html)
                plt.close(fig)
                return None
            else:
                plt.close(fig)
                raise ValueError("method must be 'writer' or 'js'.")

        # Notebook display path
        plt.close(fig)
        return IPython.display.HTML(ani.to_jshtml())
