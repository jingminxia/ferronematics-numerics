from firedrake import *
from defcon import *
from bifurcation import FerronematicsProblem
import matplotlib.pyplot as plt
import numpy

problem = FerronematicsProblem()
dc = DeflatedContinuation(problem=problem, teamsize=1, verbose=True, clear_output=False, logfiles=False)

functionals = [x[1] for x in problem.functionals()]

for functional in ["q2"]:
    def decide_stability(stab):
        print("Deciding on stability of %s" % (stab,))
        return stab[0] == 0
    dc.bifurcation_diagram(functional, fixed={"c": 1, "xi": 1}, style="ob", unstablestyle="k", decide_stability=decide_stability)
    plt.xlim((0.2,3.0))
    plt.xticks(numpy.array([0.2,1.0,2.0,3.0]))
    plt.savefig(functional + ".pdf")
    plt.clf()

