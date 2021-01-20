# -*- coding: utf-8 -*-
from firedrake import *
from defcon import *
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

omega2 = -1/2+sqrt(3)/2j
omega3 = -1/2-sqrt(3)/2j

class FerronematicsProblem(BifurcationProblem):
    def mesh(self, comm):
        self.levels = 0
        self.nviz = 0
        self.N = 50
        self.degree = 1

        base = IntervalMesh(self.N, length_or_left=-1, right=1, comm=comm)
        mh = MeshHierarchy(base, self.levels+self.nviz)

        self.mh = mh
        self.CG = FunctionSpace(mh[self.levels], "CG", self.degree)
        return mh[self.levels]

    def function_space(self, mesh):
        U = VectorFunctionSpace(mesh, "CG", self.degree, dim=2)
        V = VectorFunctionSpace(mesh, "CG", self.degree, dim=2)
        Z = MixedFunctionSpace([U,V])
        print("Z.dim(): %s %s" % (Z.dim(), [Z.sub(i).dim() for i in range(2)]))

        return Z

    def parameters(self):
        c = Constant(0)
        xi = Constant(0)
        l = Constant(0)
        return [(c, "c", r"$c$"),
                (xi, "xi", r"$\xi$"),
                (l, "l", r"$l$")]

    def homorho1(self, params):
        # the first homogeneous solution of rho
        c = params[0]
        xi = params[1]
        rho1 = pow(c/8+sqrt(c**2/64 - (1+c**2/(2*xi))**3/27),1/3) + pow(c/8-sqrt(c**2/64 - (1+c**2/(2*xi))**3/27),1/3)
        return rho1

    def homorho2(self, params):
        # the second homogeneous solution of rho
        c = params[0]
        xi = params[1]
        rho2 = pow(-c/8+sqrt(c**2/64 - (1+c**2/(2*xi))**3/27),1/3) + pow(-c/8-sqrt(c**2/64 - (1+c**2/(2*xi))**3/27),1/3)
        return rho2

    def homorho3(self, params):
        # the third homogeneous solution of rho
        c = params[0]
        xi = params[1]
        rho3 = omega3*pow(-c/8+sqrt(c**2/64 - (1+c**2/(2*xi))**3/27),1/3) + omega2*pow(-c/8-sqrt(c**2/64 - (1+c**2/(2*xi))**3/27),1/3)
        return rho3

    def energy(self, z, params):
        c = params[0]
        xi = params[1]
        l1 = params[2]
        l2 = params[2]

        (q, m) = split(z)
        E = (
                l1/2 * inner(grad(q), grad(q)) * dx
                + (inner(q, q) - 1)**2 * dx
                + xi*l2/2 * inner(grad(m), grad(m)) * dx
                + xi/4 * (inner(m, m) - 1)**2 * dx
                - c * (q[0]*(m[0]**2 - m[1]**2) + 2*q[1]*m[0]*m[1]) * dx
            )

        return E

    def residual(self, z, params, w):
        L = self.energy(z, params)
        F = derivative(L, z, w)
        return F

    def boundary_conditions(self, Z, params):
        qbc1 = DirichletBC(Z.sub(0), Constant((+1, 0)), 1)
        qbc2 = DirichletBC(Z.sub(0), Constant((-1, 0)), 2)
        mbc1 = DirichletBC(Z.sub(1), Constant((+1, 0)), 1)
        mbc2 = DirichletBC(Z.sub(1), Constant((-1, 0)), 2)
        bcs = [mbc1, mbc2, qbc1, qbc2]

        return bcs

    def functionals(self):
        def energy(z, params):
            return assemble(self.energy(z, params))

        def q1(z, params):
            c = params[0]

            (q, m) = split(z)
            j = assemble(q[0]*dx)
            return j

        def m1(z, params):
            c = params[0]

            (q, m) = split(z)
            j = assemble(m[0]*dx)
            return j

        def q2(z, params):
            c = params[0]

            (q, m) = split(z)
            j = assemble(q[1]*dx)
            return j

        return [
                (q1, "q1", r"$\int_\Omega Q_{11}$"),
                (m1, "m1", r"$\int_\Omega M_{1}$"),
                (q2, "q2", r"$\int_\Omega Q_{12}$"),
                (energy, "energy", r"$F(Q_{11}, Q_{12}, M_1, M_2)$"),
               ]

    def number_initial_guesses(self, params):
        return 1 

    def initial_guess(self, Z, params, n):
        z = Function(Z)
        x = SpatialCoordinate(Z.mesh())
        # exact solution for Laplace limit
        #z.sub(0).interpolate(as_vector([2*x[0]-1, 0]))
        #z.sub(1).interpolate(as_vector([2*x[0]-1, 0]))

        z.sub(0).interpolate(Constant((0, 1)))
        z.sub(1).interpolate(Constant((1/sqrt(2), 1/sqrt(2))))
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
            "snes_rtol": 1.0e-7,
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

    def save_figs(self, z, filename, params):
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

        (q, m) = zsrc.split()
        mnorm = Function(self.CG).interpolate(sqrt(inner(m,m)))
        m.rename("magnetization")
        q1 = q[0]
        q2 = q[1]
        Q = interpolate(as_tensor(((q1, q2), (q2, -q1))), TensorFunctionSpace(mesh, "CG", 1, shape=(2,)*2))
        eigs, eigv = np.linalg.eigh(np.array(Q.vector()))
        n = Function(VectorFunctionSpace(mesh, "CG", 1, dim=2))
        n.vector()[:,:] = eigv[:,:,1]
        n.rename("director")
        Qnorm = Function(self.CG)
        Qnorm.interpolate(sqrt(inner(Q,Q)))
        Qnorm.rename("norm-of-Q")

        c = params[0]
        xi = params[1]
#        print("Max of rho**2+sigma**2: %s" % float(max(Qnorm.dat.data_ro/2)+max(mnorm.dat.data_ro)))
#        rhomax = pow(c/8+sqrt(c**2/64- (1+c**2/(2*xi))**3/27),1/3) + pow(c/8-sqrt(c**2/64- (1+c**2/(2*xi))**3/27),1/3)
#        bound = 1+2*c/xi*rhomax + rhomax**2
#        print("Expected bound: %s" % bound)

        qnorm = Function(self.CG).interpolate(sqrt(inner(q,q)))
        print("Max of rho: %s" % max(qnorm.dat.data_ro))
        print("Max of sigma: %s" % max(mnorm.dat.data_ro))

        # render the plot
        fig = plt.figure()
        gridspec.GridSpec(4,4)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.98, bottom=0.05)
        # subplot of |Q|, Q_{11}, Q_{12}, rho^*
        plt.subplot2grid((4,4), (0,0), colspan=3, rowspan=1)
        plt.plot(coords, Qnorm.dat.data_ro/sqrt(2), 'b', label=r'$|Q|$', linewidth=4)
        plt.plot(coords, np.full(len(coords), self.homorho1(params)), '--k', label=r'$\rho^*$', linewidth=4)
        #plt.plot(coords, q.sub(0).dat.data_ro, '--g', label=r'$Q_{11}$', linewidth=4)
        #plt.plot(coords, q.sub(1).dat.data_ro, '-.c', label=r'$Q_{12}$', linewidth=4)
        plt.xticks([-1.0,-0.5,0,0.5,1.0],fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(loc="lower right", bbox_to_anchor=(0.9,0.0), frameon=False, fontsize=18)
        plt.xlabel(r'$y$', fontsize=18)
        # subplot for director
        plt.subplot2grid((4,4), (0,3), colspan=1, rowspan=1)
        plt.quiver(np.zeros(self.N+1), coords[::self.degree], np.abs(n.sub(0).dat.data), np.abs(n.sub(1).dat.data), color='k', scale=5.0, headwidth=1, headlength=1.0, headaxislength=0.01, pivot='mid')
        plt.axis("off")
        plt.title(r"$\mathbf{n}$", fontsize=18)
        theta = np.arctan2(q.sub(1).dat.data_ro, q.sub(0).dat.data_ro)
        # subplot for |M|, M_1, M_2, sigma^*
        plt.subplot2grid((4,4), (1,0), colspan=3, rowspan=1)
        plt.plot(coords, mnorm.dat.data_ro, 'r', label=r'$|M|$', linewidth=4)
        plt.plot(coords, np.full(len(coords), sqrt(1+2*c*self.homorho1(params))), '--k', label=r'$\sqrt{1+2c\rho^*}$', linewidth=4)
        #plt.plot(coords, m.sub(0).dat.data_ro, '--g', label=r'$M_1$', linewidth=4)
        #plt.plot(coords, m.sub(1).dat.data_ro, '-.c', label=r'$M_2$', linewidth=4)
        plt.legend(loc="lower right", bbox_to_anchor=(0.95,0.0), frameon=False, fontsize=18)
        plt.xlabel(r'$y$', fontsize=18)
        plt.xticks([-1.0,-0.5,0,0.5,1.0],fontsize=18)
        plt.yticks(fontsize=18)
        # subplot of M
        plt.subplot2grid((4,4), (1,3), colspan=1, rowspan=1)
        m1 = m.sub(0).dat.data_ro/(mnorm.dat.data_ro+1e-15)
        m2 = m.sub(1).dat.data_ro/(mnorm.dat.data_ro+1e-15)
        plt.quiver(np.zeros(self.N+1), coords[::self.degree], m1[::self.degree], m2[::self.degree], color='k', scale=5.0, headwidth=8, headlength=8, headaxislength=8, pivot='mid')
        plt.axis("off")
        plt.title(r'$\mathbf{m}$', fontsize=18)
        phi = np.arctan2(m.sub(1).dat.data_ro, m.sub(0).dat.data_ro)
        # subplot of 2\phi-\theta
        plt.subplot2grid((4,4), (2,0), colspan=4, rowspan=1)
        plt.plot(coords, 2*phi-theta, 'k', linewidth=4)
        plt.ylabel(r"$2\phi-\theta$", fontsize=18)
        plt.xticks([-1.0,-0.5,0,0.5,1.0],fontsize=18)
        plt.yticks([-2*pi,-pi,0,pi,2*pi],[r"$-2\pi$",r"$-\pi$",r"$0$",r"$\pi$",r"$2\pi$"],fontsize=18)
        plt.xlabel(r"$y$", fontsize=18)
        #subplot of \theta
        plt.subplot2grid((4,4), (3,0), colspan=2, rowspan=1)
        plt.plot(coords, theta, 'k', linewidth=4)
        plt.xlabel(r'$y$', fontsize=18)
        plt.xticks([-1.0,-0.5,0,0.5,1.0],fontsize=18)
        plt.yticks([-pi,-pi/2,0,pi/2,pi],[r"$-\pi$",r"$-\pi/2$",r"$0$",r"$\pi/2$",r"$\pi$"],fontsize=18)
        plt.ylabel(r'$\theta$', fontsize=18)
        # subplot of \phi
        plt.subplot2grid((4,4), (3,2), colspan=2, rowspan=1)
        plt.plot(coords, phi, 'k', linewidth=4)
        plt.xlabel(r'$y$', fontsize=18)
        plt.xticks([-1.0,-0.5,0,0.5,1.0],fontsize=18)
        plt.yticks([-pi,-pi/2,0,pi/2,pi],[r"$-\pi$",r"$-\pi/2$",r"$0$",r"$\pi/2$",r"$\pi$"],fontsize=18)
        plt.ylabel(r'$\phi$', fontsize=18, labelpad=-400)
        plt.xlabel(r"$y$", fontsize=18)
        plt.tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True)
        #fig.tight_layout()
        fig.set_size_inches(w=12,h=18)
        fig.savefig(filename)
        plt.clf()

    def monitor(self, params, branchid, solution, functionals):
        os.makedirs('./output/figs/l-%s/b-%d' % (params[2], branchid))
        filename = 'output/figs/l-%s/b-%d/solution.png' % (params[2], branchid)
        self.save_figs(solution, filename, params)
        print("Wrote to %s" % filename)
#        c = params[0]
#        xi = params[1]
#        print("The first homogeneous solution of rho: %s" % self.homorho1(params))
#        print("The first homogeneous solution of sigma: %s" % sqrt(2*c*self.homorho1(params)/xi + 1))
#        print("The second homogeneous solution of rho: %s" % self.homorho2(params))
#        print("The second homogeneous solution of sigma: %s" % sqrt(1-2*c*self.homorho2(params)/xi))
#        print("The third homogeneous solution of rho: %s" % self.homorho3(params))
#        print("The third homogeneous solution of sigma: %s" % sqrt(1-2*c*self.homorho3(params)/xi))

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
    params = linspace(0.2, 3.0, 281)
    #dc.run(values={"c": 1, "xi": 1, "l": 0.01}, freeparam="l")
    dc.run(values={"c": 5, "xi": 1, "l": 0.01}, freeparam="l")
