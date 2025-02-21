from casadi import sqrt, tanh

def actdyn_mclean2003(a, Tact, Tdeact, u):
    """
    McLean 2003 model:
    adot = (u/Tact + (1-u)/Tdeact) * (u - a)
    """
    return (u / Tact + (1 - u) / Tdeact) * (u - a)


def actdyn_mclean2003_improved(a, Tact, Tdeact, u):
    """
    McLean 2003 Improved model:
    - Activation rate depends on (u - a).
    - Avoids instability when u>1.
    """
    x = 10.0 * (u - a)
    # sigmoid-like function
    f_val = 0.5 + 0.5 * (x / sqrt(1 + x ** 2))
    # Weighted average of (1/Tact) and (1/Tdeact)
    R = f_val * (1 / Tact) + (1 - f_val) * (1 / Tdeact)
    return R * (u - a)


def actdyn_degroote2016_original(a, Tact, Tdeact, u):
    """
    De Groote et al. 2016, Original:
      b=0.1
      f = 0.5 * tanh(b*(u-a))
      adot = [ (1/Tact)/(0.5+1.5a)*(f+0.5) + (0.5+1.5a)/Tdeact*(-f+0.5) ] * (u - a)
    """
    b = 0.1
    f_val = 0.5 * tanh(b * (u - a))
    term1 = (1 / Tact) / (0.5 + 1.5 * a) * (f_val + 0.5)
    term2 = (0.5 + 1.5 * a) / Tdeact * (-f_val + 0.5)
    return (term1 + term2) * (u - a)


def actdyn_degroote2016(a, Tact, Tdeact, u):
    """
    De Groote et al. 2016, variant:
      b=10
      f = 0.5 * tanh(b*(u-a))
      adot = [ (1/Tact)/(0.5+1.5a)*(f+0.5) + (0.5+1.5a)/Tdeact*(-f+0.5) ] * (u - a)
    """
    b = 10
    f_val = 0.5 * tanh(b * (u - a))
    term1 = (1 / Tact) / (0.5 + 1.5 * a) * (f_val + 0.5)
    term2 = (0.5 + 1.5 * a) / Tdeact * (-f_val + 0.5)
    return (term1 + term2) * (u - a)