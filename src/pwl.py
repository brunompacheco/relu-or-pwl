import numpy as np
import pandas as pd

from scipy.optimize import linprog


class PWL():
    def __init__(self, inputs=['bsw', 'rgl', 'qliq', 'psup'],
                 output='delta_p'):
        self.inputs = inputs
        self.output = output

    def fit(self, df: pd.DataFrame,):
        df_ = df.round(6)  # avoid errors when comparing floats

        X = df_[self.inputs].values
        y = df_[self.output].values
        
        # unique values of x
        self.X_breakpoints = [np.sort(np.unique(x)) for x in X.T]
        self.n_X = list(map(len, self.X_breakpoints))

        self.curve = df_.set_index(self.inputs)[self.output]
    
    def __call__(self, x: np.array) -> float:
        n = len(self.n_X)  # number of variables

        # find the breakpoints around x, which form an hypercube, and the
        # position of x within this hypercube (`eta`)
        hypercube_limits, eta = self.get_hypercube_containing_x(x)

        # weight of each vertex of the hypercube.
        # theta.shape == (2**n,2)  # this vector is not actually needed

        # indices of theta in crescent order, that is, theta_indices[3] == [0,1,1] means
        # that theta[3] is associated with the vertex with x0_L, x1_H and x2_H
        theta_indices = np.mgrid[*[slice(2) for _ in range(n)]].T.reshape(-1,n)[:,::-1]

        # constraints on "high" faces
        A_H, b_H = self.get_face_constraints(eta, theta_indices, high=True)

        # constraints on "low" faces
        A_L, b_L = self.get_face_constraints(eta, theta_indices, high=False)

        assert np.all(A_L + A_H == 1)
        assert np.all(b_L + b_H == 1)

        # constraints on the output value
        A_y, b_y = self.get_output_constraints(hypercube_limits, theta_indices)

        # set up optimization
        A = np.bmat([
            [A_L,np.zeros((n,1))],
            [A_H,np.zeros((n,1))],
            [A_y,np.ones((1,1))],
        ])
        b = np.bmat([b_L, b_H, b_y]).T

        # maximize the y value
        c = np.eye(len(theta_indices) + 1)[-1]

        # all thetas have to be limited to (0,1)
        bounds = [(0.,1.),] * len(theta_indices)
        bounds.append((0,None))

        res = linprog(c, A_eq=A, b_eq=b, bounds=bounds)

        return res.x[-1]

    def get_output_constraints(self, hypercube_limits, theta_indices):
        y_vertices = list()
        for theta_index in theta_indices:
            x_vertex = np.diag(hypercube_limits.T[theta_index])
            y_vertices.append(self.curve.loc[*x_vertex])

        y_vertices = np.array(y_vertices)

        A_y = -y_vertices[None,:]
        b_y = np.array([0.,])
        return A_y,b_y

    def get_face_constraints(self, eta, theta_indices, high=True):
        A = list()
        b = list()
        for i in range(len(eta)):
            # only applies to the theta variables at the high|low face
            a = theta_indices[:,i] == int(high)
            b_i = eta[i] if high else 1 - eta[i]

            A.append(a.astype(float))
            b.append(b_i)

        A = np.stack(A)
        b = np.array(b)

        return A,b

    def get_hypercube_containing_x(self, x):
        eta = list()
        hypercube_limits = list()
        for i in range(len(x)):
            x_i_breakpoints = self.X_breakpoints[i]
            
            # faces of the hypercube of breakpoints that enclose the input
            x_i_L = x_i_breakpoints[x_i_breakpoints <= x[i]][-1]
            x_i_H = x_i_breakpoints[x_i_breakpoints > x[i]][0]
            hypercube_limits.append([x_i_L, x_i_H])

            # relative position of the input with respect to the hypercube's faces
            eta.append((x[i] - x_i_L) / (x_i_H - x_i_L))

        hypercube_limits = np.array(hypercube_limits)
        return hypercube_limits, eta
