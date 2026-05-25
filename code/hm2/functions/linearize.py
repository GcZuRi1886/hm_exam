
import sympy as sp

sp.init_printing()

def linearize_function(f, variables, point):
    """
    Linearizes the function f at a given point.
    
    Parameters:
    f (sympy.Matrix): The function to linearize.
    variables (sympy.Matrix): The variables of the function.
    point (list): The point at which to linearize the function.
    
    Returns:
    sympy.Matrix: The linearized function.
    """
    # Compute the Jacobian matrix
    Df = f.jacobian(variables)

    # Zip variables and point values into a dictionary for substitution
    substitution_dict = {var: val for var, val in zip(variables, point)}
    
    # Evaluate the Jacobian at the given point
    Df_at_point = Df.subs(substitution_dict)
    
    # Compute the linearized function
    linearized_f = f.subs(substitution_dict) + Df_at_point * (variables - sp.Matrix(point))
    
    return linearized_f

if __name__ == "__main__":
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    
    f1 = x1 + x2**2 - x3**2 -13
    f2 = sp.ln(x2**2 / 4) + sp.exp(0.5*x3 - 1) - 1
    f3 = (x2 - 3)**2 - x3**3 + 7

    f = sp.Matrix([f1, f2, f3])
    variables = sp.Matrix([x1, x2, x3])
    point = [1.5, 3, 2.5]

    linearized_f = linearize_function(f, variables, point)
    print(f"Linearized function at point {point}:")
    for row in linearized_f:
        print(row)

