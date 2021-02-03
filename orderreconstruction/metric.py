# -*- coding: utf-8 -*-
from firedrake import *
from defcon import *
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

class FerronematicsProblem(BifurcationProblem):
    def mesh(self, comm):
        self.levels = 0
        self.nviz = 0
        self.N = 1000
        self.degree = 3

        base = IntervalMesh(self.N, length_or_left=-1, right=1, comm=comm)
        mh = MeshHierarchy(base, self.levels+self.nviz)

        self.mh = mh
        self.CG = FunctionSpace(mh[self.levels], "CG", self.degree)
        return mh[self.levels]

    def function_space(self, mesh):
        U = FunctionSpace(mesh, "Hermite", self.degree)
        Z = MixedFunctionSpace([U,U])
        print("Z.dim(): %s %s" % (Z.dim(), [Z.sub(i).dim() for i in range(2)]))

        return Z

    def parameters(self):
        c = Constant(0)
        return [(c, "c", r"$c$")]

    def homorho1(self, params):
        # the first homogeneous solution of rho
        c = float(params[0])
        rho1 = pow(c/8+sqrt(c**2/64 - (1+c**2/2)**3/27),1/3) + pow(c/8-sqrt(c**2/64 - (1+c**2/2)**3/27),1/3)
        assert rho1.imag == 0
        return rho1.real

    def beta(self, params):
        # beta gives the minimum value of the OR bulk density
        c = params[0]
        q1 = self.homorho1(params)
        m1square = 1+2*c*q1
        f = (q1**2-1)**2+1/4*(m1square-1)**2-c*q1*m1square
        return f

    def energy(self, z, params):
        c = params[0]
        (q, m) = split(z)
        beta = self.beta(params)
        E = (
                ((q**2-1)**2 + 1/4*(m**2-1)**2 - c*q*m**2 - beta)*(grad(q)**2+grad(m)**2)
            )*dx

        return E

    def residual(self, z, params, w):
        L = self.energy(z, params)
        F = derivative(L, z, w)
        return F

    def boundary_conditions(self, Z, params):
        rhostar = self.homorho1(params)
        c = params[0]
        # d(p*, p**)
        qbc = DirichletBC(Z.sub(0), Constant(rhostar), [1,2])
        mbc1 = DirichletBC(Z.sub(1), Constant(sqrt(1+2*c*rhostar)), 1)
        mbc2 = DirichletBC(Z.sub(1), Constant(-sqrt(1+2*c*rhostar)), 2)
        bcs = [qbc, mbc1, mbc2]
        # d(p*, p_b(1))
        #qbc1 = DirichletBC(Z.sub(0), Constant(rhostar), 1)
        #qbc2 = DirichletBC(Z.sub(0), Constant(-1), 2)
        #mbc1 = DirichletBC(Z.sub(1), Constant(sqrt(1+2*c*rhostar)), 1)
        #mbc2 = DirichletBC(Z.sub(1), Constant(-1), 2)
        #bcs = [qbc1, qbc2, mbc1, mbc2]
        ## d(p**, p_b(-1))
        #qbc1 = DirichletBC(Z.sub(0), Constant(rhostar), 1)
        #qbc2 = DirichletBC(Z.sub(0), Constant(1), 2)
        #mbc1 = DirichletBC(Z.sub(1), Constant(-sqrt(1+2*c*rhostar)), 1)
        #mbc2 = DirichletBC(Z.sub(1), Constant(1), 2)
        #bcs = [qbc1, qbc2, mbc1, mbc2]

        return bcs

    def functionals(self):
        def energy(z, params):
            j = sqrt(assemble(self.energy(z, params)))
            print("Energy cost: %s" % j)
            return j

        def q1(z, params):
            c = params[0]

            (q, m) = split(z)
            j = assemble(q*dx)
            return j

        def m1(z, params):
            c = params[0]

            (q, m) = split(z)
            j = assemble(m*dx)
            return j

        return [
                (q1, "q1", r"$\int_\Omega Q_{11}$"),
                (m1, "m1", r"$\int_\Omega M_{1}$"),
                (energy, "energy", "energy"),
               ]

    def number_initial_guesses(self, params):
        return 1 

    def initial_guess(self, Z, params, n):
        z = Function(Z)
        x = SpatialCoordinate(Z.mesh())
        lu = {"ksp_type": "preonly",
              "pc_type": "lu",
              "mat_type": "aij",
              "pc_factor_mat_solver_type": "mumps",
              "mat_mumps_icntl_14": 200,}

        z.sub(0).project(Constant(0.5), solver_parameters=lu)
        z.sub(1).project(Constant(0.5), solver_parameters=lu)
        return z

    def number_solutions(self, params):
        return float("inf")

    def solver_parameters(self, params, task, **kwargs):
        if isinstance(task, DeflationTask):
            damping = 0.9
            maxits = 1000
        else:
            damping = 0.9 
            maxits = 1000

        params = {
            "snes_max_it": maxits,
            "snes_rtol": 1.0e-6,
            "snes_atol": 1.0e-8,
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

    def save_figs(self, z, filename):
        zsrc = z
        mesh = z.function_space().mesh()
        x = SpatialCoordinate(mesh)
        coords = Function(self.CG).interpolate(x[0]).dat.data_ro

        if self.nviz > 0:
            ele = z.function_space().ufl_element()
            transfer = TransferManager()
            for i in range(self.nviz):
                mesh = self.mh[self.levels + i + 1]
                Z = FunctionSpace(mesh, ele)
                znew = Function(Z)
                transfer.prolong(zsrc, znew)
                zsrc = znew

        (q1, m1) = zsrc.split()
        q = project(q1, self.CG)
        m = project(m1, self.CG)
        # render the plot
        plt.figure()
        plt.plot(coords, q.dat.data_ro, '-b', label=r"$Q_{11}$", linewidth=2)
        plt.plot(coords, m.dat.data_ro, '-k', label=r"$M_1$", linewidth=2)
        plt.legend(loc="best", bbox_to_anchor=(0.2,0.2), frameon=False, fontsize=8)
        plt.xlabel(r'$y$')
        plt.savefig(filename)
        plt.clf()

    def monitor(self, params, branchid, solution, functionals):
        os.makedirs('./output/figs/c-%s/b-%d' % (params[0], branchid))
        filename = 'output/figs/c-%s/b-%d/solution.png' % (params[0], branchid)
        self.save_figs(solution, filename)
        print("Wrote to %s" % filename)

    def compute_stability(self, params, branchid, z, hint=None):
        Z = z.function_space()
        trial = TrialFunction(Z)
        test  = TestFunction(Z)

        bcs = self.boundary_conditions(Z, params)
        comm = Z.mesh().mpi_comm()

        F = self.residual(z, [Constant(p) for p in params], test)
        J = derivative(F, z, trial)

        # Build the LHS matrix
        A = assemble(J, bcs=bcs, mat_type="aij")
        A = A.M.handle

        pc = PETSc.PC().create(comm)
        pc.setOperators(A)
        pc.setType("cholesky")
        pc.setFactorSolverType("mumps")
        pc.setUp()

        F = pc.getFactorMatrix()
        (neg, zero, pos) = F.getInertia()

        print("Inertia: (-: %s, 0: %s, +: %s)" % (neg, zero, pos))
        expected_dim = 0

        # Nocedal & Wright, theorem 16.3
        if neg == expected_dim:
            is_stable = True
        else:
            is_stable = False

        d = {"stable": (neg, zero, pos)}
        return d

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=FerronematicsProblem(), teamsize=1, verbose=True, clear_output=True, logfiles=False)
    params = linspace(1, 3, 201)
    dc.run(values={"c": params[0]}, freeparam="c")
