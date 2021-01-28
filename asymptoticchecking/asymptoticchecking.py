# -*- coding: utf-8 -*-
from firedrake import *
from defcon import *
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
import os

# we set l1=l2=1/c later
xi = Constant(1)
c = Constant(0.1)

class FerronematicsProblem(object):
    def __init__(self):
        super().__init__()

    def mesh(self):
        self.levels = 0
        self.nviz = 0
        self.N = 1000

        base = IntervalMesh(self.N, length_or_left=-1, right=1)
        mh = MeshHierarchy(base, self.levels+self.nviz)

        self.mh = mh
        self.CG1 = FunctionSpace(mh[self.levels], "CG", 1)
        return mh[self.levels]

    def function_space(self, mesh):
        U = VectorFunctionSpace(mesh, "CG", 1, dim=2)
        V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
        Z = MixedFunctionSpace([U,V])
        print("Z.dim(): %s %s" % (Z.dim(), [Z.sub(i).dim() for i in range(2)]))

        return Z

    def energy(self, z):
        (q, m) = split(z)
        E = (
                1/(2*c) * inner(grad(q), grad(q)) * dx
                + (inner(q, q) - 1)**2 * dx
                + xi/(2*c) * inner(grad(m), grad(m)) * dx
                + xi/4 * (inner(m, m) - 1)**2 * dx
                - c * (q[0]*(m[0]**2 - m[1]**2) + 2*q[1]*m[0]*m[1]) * dx
            )

        return E

    def boundary_conditions(self, Z):
        qbc1 = DirichletBC(Z.sub(0), Constant((+1, 0)), 1)
        qbc2 = DirichletBC(Z.sub(0), Constant((-1, 0)), 2)
        mbc1 = DirichletBC(Z.sub(1), Constant((+1, 0)), 1)
        mbc2 = DirichletBC(Z.sub(1), Constant((-1, 0)), 2)
        bcs = [mbc1, mbc2, qbc1, qbc2]

        return bcs

    def initial_guess(self, z):
        x = SpatialCoordinate(z.function_space().mesh())
        # exact solution for Laplace limit
        #z.sub(0).interpolate(as_vector([-x[0], 0]))
        #z.sub(1).interpolate(as_vector([-x[0], 0]))

        z.sub(0).interpolate(Constant((0, 1)))
        z.sub(1).interpolate(Constant((1/sqrt(2), 1/sqrt(2))))
        return z

    def asymptotic_q(self, mesh):
        x = SpatialCoordinate(mesh)
        q1_asymp = ( -x[0]
                     #+ c * (-1/5*pow(x[0],5) + 2/3*pow(x[0],3) - 7/15*x[0])
                     #+ c**2 * (-1/30*pow(x[0],9) + 22/105*pow(x[0],7) - 31/75*pow(x[0],5) - 1/12*pow(x[0],4) + 14/45*pow(x[0],3) - 233/3150*x[0] + 1/12)
                   )
        return as_vector([q1_asymp, 0])

    def asymptotic_m(self, mesh):
        x = SpatialCoordinate(mesh)
        m1_asymp = ( -x[0]
                     #+ c * (-1/20*pow(x[0],5) + 1/6*pow(x[0],3) - 7/60*x[0])
                     #+ c**2 * (-1/480*pow(x[0],9) + 11/840*pow(x[0],7) - 31/1200*pow(x[0],5) + 7/360*pow(x[0],3) - 1/(6*xi)*pow(x[0],4) - 233/50400*x[0]+1/(6*xi))
                   )
        return as_vector([m1_asymp, 0])

    def solver_parameters(self):
        damping = 0.9
        maxits = 500

        params = {
            "snes_max_it": maxits,
            "snes_rtol": 1.0e-20,
            "snes_atol": 1.0e-6,
            "snes_stol":    0.0,
            "snes_monitor": None,
            "snes_linesearch_type": "l2",
            "snes_linesearch_monitor": None,
            "snes_linesearch_maxstep": 1.0,
            "snes_linesearch_damping": damping,
            "snes_converged_reason": None,
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 200,
            "mat_mumps_icntl_24": 1,
            "mat_mumps_icntl_13": 1
        }
        return params

    def save_figs(self, z):
        os.makedirs('./output/figs/c-%s' % c, exist_ok=True)
        filename = 'output/figs/c-%s/solution.png' % c
        zsrc = z
        mesh = z.function_space().mesh()
        x = SpatialCoordinate(mesh)
        coords = Function(self.CG1).interpolate(x[0]).dat.data_ro

        if self.nviz > 0:
            ele = z.function_space().ufl_element()
            transfer = TransferManager()
            for i in range(self.nviz):
                mesh = self.mh[self.levels + i + 1]
                Z = FunctionSpace(mesh, ele)
                znew = Function(Z)
                transfer.prolong(zsrc, znew)
                zsrc = znew

        (q, m) = zsrc.split()
        mnorm = Function(self.CG1).interpolate(sqrt(inner(m,m)))
        m.rename("magnetization")
        q1 = q[0]
        q2 = q[1]
        Q = interpolate(as_tensor(((q1, q2), (q2, -q1))), TensorFunctionSpace(mesh, "CG", 1, shape=(2,)*2))
        eigs, eigv = np.linalg.eigh(np.array(Q.vector()))
        n = Function(VectorFunctionSpace(mesh, "CG", 1, dim=2))
        n.vector()[:,:] = eigv[:,:,1]
        n.rename("director")
        Qnorm = Function(self.CG1)
        Qnorm.interpolate(sqrt(inner(Q,Q)))
        Qnorm.rename("norm-of-Q")

        # render the plot
        fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(2,2)
        plt.subplots_adjust(wspace=0.4, hspace=0.5)
        ax1.plot(coords, Qnorm.dat.data_ro, 'b', label=r'$|Q|$')
        ax1.plot(coords, q.sub(0).dat.data_ro, '--g', label=r'$Q_{11}$')
        ax1.plot(coords, q.sub(1).dat.data_ro, '-.c', label=r'$Q_{12}$')
        ax1.legend(loc="lower right", frameon=False, fontsize=8)
        ax1.set_xlabel(r'$y$')
        ax2.plot(coords, mnorm.dat.data_ro, 'r', label=r'$|M|$')
        ax2.plot(coords, m.sub(0).dat.data_ro, '--g', label=r'$M_1$')
        ax2.plot(coords, m.sub(1).dat.data_ro, '-.c', label=r'$M_2$')
        ax2.legend(loc="lower right", frameon=False, fontsize=8)
        ax2.set_xlabel(r'$y$')
        ax3.quiver(np.zeros(self.N+1), coords, np.abs(n.sub(0).dat.data), np.abs(n.sub(1).dat.data), color='b', headwidth=0.1, headlength=0.1, headaxislength=0.00001, pivot='mid')
        ax3.set_title('director')
        #normalize m
        m1 = m.sub(0).dat.data_ro/(mnorm.dat.data_ro+1e-10)
        m2 = m.sub(1).dat.data_ro/(mnorm.dat.data_ro+1e-10)
        ax4.quiver(np.zeros(self.N+1), coords, m1, m2, color='r', pivot='mid')
        ax4.set_title('magnetization')
        plt.savefig(filename)
        plt.clf()

class Ferronematicssolver(object):
    def __init__(self, problem):
        self.problem = problem
        self.c = c

    def solve(self, c):
        print(GREEN % ("Solving for c = %s" % c))
        self.c.assign(c)

        try:
            problem = self.problem
            mesh = problem.mesh()
            Z = problem.function_space(mesh)
            z = Function(Z)
            problem.initial_guess(z)

            L = problem.energy(z)
            v = TestFunction(Z)
            F = derivative(L, z, v)

            nvproblem = NonlinearVariationalProblem(F, z, bcs=problem.boundary_conditions(Z))
            nvsolver  = NonlinearVariationalSolver(nvproblem, solver_parameters=problem.solver_parameters())
            nvsolver.solve()

            (q_, m_) = z.split()
            problem.save_figs(z)
            CG1 = VectorFunctionSpace(mesh, "CG", 1, dim=2)
            q_asymp = Function(CG1).interpolate(problem.asymptotic_q(mesh))
            q_rem = max(abs(np.subtract(q_asymp.dat.data[:,0], q_.dat.data[:,0])))
            m_asymp = Function(CG1).interpolate(problem.asymptotic_m(mesh))
            m_rem = max(abs(np.subtract(m_asymp.dat.data[:,0], m_.dat.data[:,0])))
            #q_rem = errornorm(q_asymp, q_, norm_type="L2")
            #m_rem = errornorm(m_asymp, m_, norm_type="L2")
        except ConvergenceError:
            print("Warning: the solver did not converge at c=%s" % c)
            q_rem = 0
            m_rem = 0

        info_dict = {
                "c": c,
                "q1_rem": q_rem,# q1_rem-9.799639189505105e-3, #subtract the error from discretization
                "m1_rem": m_rem,#-6.899291343225612e-3,#m1_rem-9.799639189504994e-3, #subtract the error from discretization
                }
        return (z, info_dict)

if __name__ == "__main__":

    problem = FerronematicsProblem()
    solver = Ferronematicssolver(problem)

    start = 1e-9
    end = 1e-8
    step = 1e-9
    #clist = list(np.arange(start,end,step)) +[end]
    #clist = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    clist = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    results = []
    for c in clist:
        (z, info_dict) = solver.solve(c)
        results.append(info_dict)

    c = [d["c"] for d in results]
    q1_rem = [d["q1_rem"] for d in results]
    m1_rem = [d["m1_rem"] for d in results]
    c3 = np.power(c, 3)
    
    print("q1_rem: %s" % q1_rem)
    print("m1_rem: %s" % m1_rem)

    from math import log

    plt.figure(1)
    plt.xlabel(r"$c$")
    plt.ylabel(r"$||Q^{num}_{11}- Q^{asymp}_{11}||_{L^\infty}$")
    plt.xticks(c)
    plt.loglog(c, q1_rem, 'b', marker="s", markersize=8, linewidth=2.0)
    plt.loglog(c[0:2:1], [q1_rem[0],q1_rem[0]], "k")
    plt.loglog([c[1],c[1]], q1_rem[0:2:1], "k")
    plt.loglog(c[0:2:1], q1_rem[0:2:1], "k")
    slope_q = (log(q1_rem[1])-log(q1_rem[0]))/(log(c[1])-log(c[0]))
    plt.text(1.6e-2, 3e-4, "slope %s" % slope_q, horizontalalignment="center", verticalalignment="center") # for 0th-order truncation
    #plt.text(1.6e-2, 8e-7, "slope %s" % slope_q, horizontalalignment="center", verticalalignment="center") # for 1st-order truncation
    #plt.text(1.6e-2, 9e-10, "slope %s" % slope_q, horizontalalignment="center", verticalalignment="center") # for 2nd-order truncation
    plt.savefig("q_asymptotic_rate.pdf", bbox_inches='tight')
    plt.clf()

    plt.figure(2)
    plt.xlabel(r"$c$")
    plt.ylabel(r"$||M^{num}_{1}- M_1^{asymp}||_{L^\infty}$")
    plt.xticks(c)
    plt.loglog(c, m1_rem, 'b', marker="s", markersize=8,linewidth=2.0)
    plt.loglog(c[0:2:1], [m1_rem[0],m1_rem[0]], "k")
    plt.loglog([c[1],c[1]], m1_rem[0:2:1], "k")
    plt.loglog(c[0:2:1], m1_rem[0:2:1], "k")
    slope_m = (log(m1_rem[1])-log(m1_rem[0]))/(log(c[1])-log(c[0]))
    plt.text(1.6e-2, 8e-5, "slope %s" % slope_m, horizontalalignment="center", verticalalignment="center") # for 0th-order truncation
    #plt.text(1.6e-2, 8e-7, "slope %s" % slope_m, horizontalalignment="center", verticalalignment="center") # for 1st-order truncation
    #plt.text(1.6e-2, 9e-10, "slope %s" % slope_m, horizontalalignment="center", verticalalignment="center") # for 2nd-order truncation
    plt.savefig("m_asymptotic_rate.pdf", bbox_inches='tight')
    plt.clf()
