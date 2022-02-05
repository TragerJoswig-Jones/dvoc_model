
def calculate_power(v, i):
    p = 1.5 * (v.alpha * i.alpha + v.beta * i.beta)
    q = 1.5 * (v.beta * i.alpha - v.alpha * i.beta)
    return p, q

def calculate_current(v, p, q):
    i_alpha = 2 * (p * v.alpha + q * v.beta) / (v.alpha ** 2 + v.beta ** 2) / 3
    i_beta = 2 * (p * v.beta - q * v.alpha) / (v.alpha ** 2 + v.beta ** 2) / 3
    return i_alpha, i_beta