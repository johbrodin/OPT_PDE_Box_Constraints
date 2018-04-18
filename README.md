# OPT_PDE_Box_Constraints
This is a note on the changes made to Anders Stroms code by Johanna Brodin.

The original code can be found on github in the repository 
http://github.com/abstrom/OPT_PDE_Box_Constraints
and the modified code here in
http://github.com/johbrodin/OPT_PDE_Box_Constraints

In the modified code, changes are marked with "//Added:" and "//Commented:" 
to show where the code deviates from the original.

This version of the code has an additional stopping criterion for the nonlinear solver (implemented for the upper bound). The Full Newton has a second option for the desired state function, an extended parameter file and an optional linear solver (if one wants to take a look at unpreconditioned minres). 
