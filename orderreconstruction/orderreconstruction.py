# -*- coding: utf-8 -*-
from firedrake import *
from defcon import *
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
import os
from slepc4py import SLEPc

omega2 = -1/2+sqrt(3)/2j
omega3 = -1/2-sqrt(3)/2j

class FerronematicsProblem(BifurcationProblem):
    def mesh(self, comm):
        self.levels = 0
        self.nviz = 0
        self.N = 1000

        base = IntervalMesh(self.N, length_or_left=-1, right=1, comm=comm)
        mh = MeshHierarchy(base, self.levels+self.nviz)

        self.mh = mh
        self.degree = 1
        self.CG = FunctionSpace(mh[self.levels], "CG", self.degree)
        return mh[self.levels]

    def function_space(self, mesh):
        U = FunctionSpace(mesh, "CG", self.degree)
        V = FunctionSpace(mesh, "CG", self.degree)
        Z = MixedFunctionSpace([U,V])
        print("Z.dim(): %s %s" % (Z.dim(), [Z.sub(i).dim() for i in range(2)]))
        self.fs = MixedFunctionSpace([U,U,U,U])

        return Z

    def parameters(self):
        c = Constant(0)
        xi = Constant(0)
        l = Constant(0)
        return [(c, "c", r"$c$"),
                (xi, "xi", r"$\xi$"),
                (l, "l", r"$l$")]

    def homoQ1(self, params):
        # the first homogeneous solution of Q11
        c = params[0]
        xi = params[1]
        homoQ1 = pow(c/8+sqrt(c**2/64 - (1+c**2/(2*xi))**3/27),1/3) + pow(c/8-sqrt(c**2/64 - (1+c**2/(2*xi))**3/27),1/3)
        return homoQ1

    def homoQ2(self, params):
        # the second homogeneous solution of Q11
        c = params[0]
        xi = params[1]
        homoQ2 = omega2*pow(c/8+sqrt(c**2/64 - (1+c**2/(2*xi))**3/27),1/3) + omega3*pow(c/8-sqrt(c**2/64 - (1+c**2/(2*xi))**3/27),1/3)
        return homoQ2

    def homoQ3(self, params):
        # the third homogeneous solution of Q11
        c = params[0]
        xi = params[1]
        homoQ3 = omega3*pow(c/8+sqrt(c**2/64 - (1+c**2/(2*xi))**3/27),1/3) + omega2*pow(c/8-sqrt(c**2/64 - (1+c**2/(2*xi))**3/27),1/3)
        return homoQ3

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
                - c * q * m**2 * dx
            )

        return E

    def enlarge_energy(self, enlarge_z, params):
        c = params[0]
        xi = params[1]
        l1 = params[2]
        l2 = params[2]

        (q1, q2, m1, m2) = split(enlarge_z)
        eE = (
                l1/2 * (inner(grad(q1), grad(q1)) + inner(grad(q2), grad(q2)) )* dx
                + (q1**2 + q2**2 - 1)**2 * dx
                + xi*l2/2 * (inner(grad(m1), grad(m1)) + inner(grad(m2), grad(m2)) )* dx
                + xi/4 * (m1**2 + m2**2 - 1)**2 * dx
                - c * q1 * (m1**2 - m2**2) * dx
                - 2*c * q2 * m1 * m2 * dx
            )

        return eE

    def residual(self, z, params, w):
        L = self.energy(z, params)
        F = derivative(L, z, w)
        return F

    def enlarge_residual(self, enlarge_z, params, enlarge_w):
        eL = self.enlarge_energy(enlarge_z, params)
        eF = derivative(eL, enlarge_z, enlarge_w)
        return eF

    def boundary_conditions(self, Z, params):
        qbc1 = DirichletBC(Z.sub(0), Constant(+1), 1)
        qbc2 = DirichletBC(Z.sub(0), Constant(-1), 2)
        mbc1 = DirichletBC(Z.sub(1), Constant(+1), 1)
        mbc2 = DirichletBC(Z.sub(1), Constant(-1), 2)
        bcs = [mbc1, mbc2, qbc1, qbc2]

        return bcs

    def functionals(self):
        def energy(z, params):
            return assemble(self.energy(z, params))

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
                (energy, "energy", r"$E(Q_{11}, M_1)$"),
               ]

    def number_initial_guesses(self, params):
        return 1 

    def initial_guess(self, Z, params, n):
        z = Function(Z)
        x = SpatialCoordinate(Z.mesh())
        #z.sub(0).interpolate(Constant(1.0))
        z.sub(0).assign(self.homoQ1(params))
        #z.sub(1).assign(sqrt(1+2*params[0]*self.homoQ1(params)))
        #z.sub(1).assign(-sqrt(1+2*params[0]*self.homoQ1(params)))
        #z.sub(1).interpolate(Constant(1.0))
        z.sub(1).interpolate(Constant(-1.0))

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

    def save_figs(self, z, filename, params, branchid):
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
        q1 = q
        Q = interpolate(as_tensor(((q1, 0), (0, -q1))), TensorFunctionSpace(mesh, "CG", 1, shape=(2,)*2))
        eigs, eigv = np.linalg.eigh(np.array(Q.vector()))
        n = Function(VectorFunctionSpace(mesh, "CG", 1, dim=2))
        n.vector()[:,:] = eigv[:,:,1]
        n.rename("director")
        n1 = n.sub(0).dat.data_ro
        n2 = n.sub(1).dat.data_ro

        #with open('c-%s-xi-%s-l-%s-b-%s.txt' % (params[0],params[1],params[2],branchid), 'w') as output:
        #    for (coordvalue,q1value,m1value) in zip(coorddata,qdata,mdata):
        #        output.write(str((coordvalue,q1value,m1value)) + '\n')

        # render the plot
        fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(2,2,gridspec_kw={'width_ratios': [3,1]})
        plt.subplots_adjust(wspace=0, hspace=0.5)
        ax1.plot(coords, q.dat.data_ro, '-k', label=r"$Q_{11}$", linewidth=3)
        ax1.plot(coords, np.full(len(coords), self.homoQ1(params)), '-b', linewidth=3, label=r"$\pm\rho^*$")
        ax1.plot(coords, np.full(len(coords), -self.homoQ1(params)), '-b', linewidth=3)
        ax1.legend(loc="best", bbox_to_anchor=(0.3,0.5), frameon=False, fontsize=8)
        ax1.set_xlabel(r'$y$')
        ax2.quiver(np.zeros(self.N+1)[::50], coords[::50], np.abs(n1[::50]), np.abs(n2[::50]), color='k', scale=8.0, headwidth=0.4, headlength=0.3, headaxislength=0.01, pivot='mid')
        ax2.axis("off")
        ax2.set_title(r"$\mathbf{n}$")
        ax3.plot(coords, m.dat.data_ro, '-k', label=r"$M_1$", linewidth=3)
        c = params[0]
        ax3.plot(coords, np.full(len(coords),sqrt(1+2*c*self.homoQ1(params))), '-b', linewidth=3, label=r"$\pm\sqrt{1+2c\rho^*}$")
        ax3.plot(coords, np.full(len(coords),-sqrt(1+2*c*self.homoQ1(params))), '-b', linewidth=3)
        ax3.legend(loc="lower left", bbox_to_anchor=(0.1,0.1), frameon=False, fontsize=8)
        ax3.set_xlabel(r'$y$')
        m1 = m.dat.data_ro/(np.sqrt(m.dat.data_ro**2)+1e-15)
        size = len(m1)
        m2 = np.zeros(size, dtype=int)
        ax4.quiver(np.zeros(self.N+1)[::50], coords[::50], m1[::50], m2[::50], color='k', scale=8.0, headwidth=5, headlength=7, headaxislength=7, pivot='mid')
        ax4.axis("off")
        ax4.set_title(r'$\mathbf{m}$')
        plt.savefig(filename)
        plt.clf()

    def monitor(self, params, branchid, solution, functionals):
        os.makedirs('./output/figs/l-%s/b-%d' % (params[2], branchid))
        filename = 'output/figs/l-%s/b-%d/solution.png' % (params[2], branchid)
        c = params[0]
        xi = params[1]
        self.save_figs(solution, filename, params, branchid)
        print("Wrote to %s" % filename)
        print("The first homogeneous solution of Q11: %s" % self.homoQ1(params))
        print("The first homogeneous solution of M1^2: %s" % (2*c*self.homoQ1(params)/xi + 1))
        print("The second homogeneous solution of Q11: %s" % self.homoQ2(params))
        print("The second homogeneous solution of M1^2: %s" % (2*c*self.homoQ2(params)/xi + 1))
        print("The third homogeneous solution of Q11: %s" % self.homoQ3(params))
        print("The third homogeneous solution of M1^2: %s" % (2*c*self.homoQ3(params)/xi + 1))

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

        print("Inertia under OR functional: (-: %s, 0: %s, +: %s)" % (neg, zero, pos))
        expected_dim = 0

        # Nocedal & Wright, theorem 16.3
        if neg == expected_dim:
            is_stable = True
        else:
            is_stable = False
        d = {"stable": (neg, zero, pos)}

        #compute stability under enlarged functional
        self.compute_enlarged_stability(params, branchid, z)
        return d

    def compute_enlarged_stability(self, params, branchid, z, hint=None):
        fs = self.fs
        Z = z.function_space()
        trial = TrialFunction(fs)
        test  = TestFunction(fs)
        enlarge_z = Function(fs)
        enlarge_z.sub(0).interpolate(z.sub(0))
        enlarge_z.sub(1).interpolate(Constant(0))
        enlarge_z.sub(2).interpolate(z.sub(1))
        enlarge_z.sub(3).interpolate(Constant(0))

        q1bc1 = DirichletBC(fs.sub(0), Constant(+1), 1)
        q1bc2 = DirichletBC(fs.sub(0), Constant(-1), 2)
        m1bc1 = DirichletBC(fs.sub(2), Constant(+1), 1)
        m1bc2 = DirichletBC(fs.sub(2), Constant(-1), 2)
        q2bc = DirichletBC(fs.sub(1), Constant(0), "on_boundary")
        m2bc = DirichletBC(fs.sub(3), Constant(0), "on_boundary")
        bcs = [q1bc1, q1bc2, q2bc, m1bc1, m1bc2, m2bc]

        comm = Z.mesh().mpi_comm()

        F = self.enlarge_residual(enlarge_z, [Constant(p) for p in params], test)
        J = derivative(F, enlarge_z, trial)

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

        print("Inertia under enlarged functional: (-: %s, 0: %s, +: %s)" % (neg, zero, pos))
        expected_dim = 0

        # Nocedal & Wright, theorem 16.3
        if neg == expected_dim:
            is_stable = True
        else:
            is_stable = False

            # Compute eigenfunction for the unstable case
            print("start computing eigenvalues for unstable eigenvalues...")
            M = inner(test, trial)*dx
            M = assemble(M, bcs=bcs, mat_type="aij")
            M = M.M.handle
            # Zero the rows and columns of M associated with bcs:
            from firedrake.preconditioners.patch import bcdofs
            lgmap = fs.dof_dset.lgmap
            for bc in bcs:
                M.zeroRowsColumns(lgmap.apply(bcdofs(bc)), diag=0)

            solver_parameters = {
                                "eps_gen_hermitian": None,
                                "eps_type": "krylovschur",
                                "eps_monitor_conv": None,
                                "eps_smallest_magnitude": None,
                                "eps_target": -1,
                                "st_type": "sinvert",
                                }
            opts = PETSc.Options()
            for (key, val) in solver_parameters.items():
                opts[key] = val

            # Create the SLEPc eigensolver
            eps = SLEPc.EPS().create(comm=comm)
            eps.setOperators(A, M)
            eps.setFromOptions()
            eps.solve()

            eigenvalues = []
            eigenfunctions = []
            for i in range(eps.getConverged()):
                eigenvalue = eps.getEigenvalue(i)
                assert eigenvalue.imag == 0
                eigenvalues.append(eigenvalue.real)
                eigenfunction = Function(fs, name="Eigenfunction")
                with eigenfunction.dat.vec_wo as vec:
                    eps.getEigenvector(i, vec)
                eigenfunctions.append(eigenfunction.copy(deepcopy=True))
            # save eigenfunctions
            self.save_eigenfunctions(eigenvalues, eigenfunctions, params, Z, branchid)
            print("Minimal eigenvalue: %s" % min(eigenvalues))

    def save_eigenfunctions(self, eigenvalues, eigenfunctions, params, Z, branchid):
        os.makedirs('./output/eigenfunctions/l-%s-b-%s' % (params[2], branchid), exist_ok=True)
        mesh = Z.mesh()
        x = SpatialCoordinate(mesh)
        coords = Function(FunctionSpace(mesh, "CG", self.degree)).interpolate(x[0]).dat.data_ro
        num = 1
        for (eigenvalue, eigenfunction) in zip(eigenvalues, eigenfunctions):
            print("Got eigenvalue %s" % eigenvalue)
            (q1, q2, m1, m2) = eigenfunction.split()
            fig, (ax1,ax2) = plt.subplots(2,1)
            plt.subplots_adjust(left = +0.16, wspace=0.4, hspace=0.5)
            ax1.plot(coords, q2.dat.data_ro, '.k', label=r'$Q_{12}$')
            #ax1.legend(loc="lower right", frameon=False, fontsize=8)
            ax1.set_xlabel(r'$y$')
            ax1.set_ylabel(r'$Q_{12}$')
            ax2.plot(coords, m2.dat.data_ro, '.k', label=r'$M_2$')
            #ax2.legend(loc="lower right", frameon=False, fontsize=8)
            ax2.set_xlabel(r'$y$')
            ax2.set_ylabel(r'$M_2$')
            filename = "output/eigenfunctions/l-%s-b-%s/eig-%s.png" % (params[2], branchid, num)
            plt.savefig(filename)
            plt.clf()
            print("Saved eigenfunctions to %s." % filename)
            num += 1
            plt.close('all')

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=FerronematicsProblem(), teamsize=1, verbose=True, clear_output=True, logfiles=False)
    params = linspace(0.0001, 0.01, 100)
    #dc.run(values={"c": 1, "xi": 1, "l": 10}, freeparam="l")
    #dc.run(values={"c": 1, "xi": 1, "l": 0.01}, freeparam="l")
    #dc.run(values={"c": 5, "xi": 1, "l": 0.01}, freeparam="l")
    dc.run(values={"c": 1, "xi": 1, "l": 1}, freeparam="l")
