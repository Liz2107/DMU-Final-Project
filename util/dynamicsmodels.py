import numpy as np

def eom_cr3bp(_, y, mu):
    r = y[0:3]
    v = y[3:6]

    x = r[0]
    y = r[1]
    z = r[2]

    vx = v[0]
    vy = v[1]

    r_mu = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r_ommu = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)
    
    dVdx = x - (1 - mu) / r_mu ** 3 * (x + mu) - mu / r_ommu ** 3 * (x - 1 + mu)
    dVdy = y - (1 - mu) / r_mu ** 3 * y - mu / r_ommu ** 3 * y
    dVdz = -(1 - mu) / r_mu ** 3 * z - mu / r_ommu ** 3 * z

    ax = 2 * vy + dVdx
    ay = -2 * vx + dVdy
    az = dVdz

    drdt = v
    dvdt = np.array([ax, ay, az])

    return np.concat((drdt, dvdt))
