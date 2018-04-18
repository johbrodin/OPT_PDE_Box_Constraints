# OPT_PDE_Box_Constraints
This is a note on the changes made to Anders Stroms code by Johanna Brodin.

The original code can be found on github in the repository 
http://github.com/abstrom/OPT_PDE_Box_Constraints
and the modified code here in
http://github.com/johbrodin/OPT_PDE_Box_Constraints

In the modified code, changes are marked with "//Added:" and "//Commented:" 
to show where the code deviates from the original.
Some changes are:
for the Full newton:
	Added a second option for the desired state function
	Added more parameters for use in the parameter file
	Added minres as an optional solver (for testing)
	Added an extra NL stopping criterion (implemented for the upper bound only)

for the Hybrid Newton: 
	Added an extra NL stopping criterion (implemented for the upper bound only)
