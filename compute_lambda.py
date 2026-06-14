#!/usr/bin/env python3
"""Read PLUMED HILLS: mean |delta CV| between successive hills (periodic if torsion)."""

from pathlib import Path
import numpy as np

#USER INPUTS ----->
# First column of the HILLS is time; bucket CV is in column 2, thus setting CV_COL=1
CV_COL = 1
cv_is_torsion= True # Make it False if the CV is not a torision

ETA_L=np.array([-3.14, -1.05, 1.05]) #\eta_L
ETA_U=np.array([-1.05,  1.05, 3.14]) #\eta_u
#END OF USER INPUTS

print("INPUTS:")
print("Column Number of HILLS containing the bucket CV=", CV_COL)
if cv_is_torsion:
   print("Selected CV is a Torsion")
else:
   print("Selected CV is a _not_ a Torsion")

print("eta_lower= ", ETA_L)
print("eta_upper= ", ETA_U)

PI = np.pi
TWO_PI = 2 * np.pi


def wrapped_abs_diff(cv_new, cv_old):
    diff = cv_new - cv_old
    if cv_is_torsion:
        diff_pbc = diff - TWO_PI * np.round(diff / TWO_PI)  # comment if bucket CV is not a torsion
    else:
        diff_pbc = diff 
    return abs(diff_pbc)


def compute_lambda():
    hills_path = Path("HILLS")
    if not hills_path.is_file():
        raise SystemExit(f"not found: {hills_path.resolve()}")
    lines = hills_path.read_text().splitlines()
    cv: list[float] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cv.append(float(line.split()[CV_COL]))
    n = len(cv)
    if n < 2:
        raise SystemExit("need at least two hill values after skipping comments")
    total = sum(wrapped_abs_diff(cv[i], cv[i - 1]) for i in range(1, n))
    lam=total/(n-1) #Lambda

    xi_L = ETA_L - 0.5* lam  
    xi_U = ETA_U + 0.5* lam
    
    if cv_is_torsion:
    	xi_Lpbc = xi_L - TWO_PI * np.round(xi_L / TWO_PI)
    	xi_Upbc = xi_U - TWO_PI * np.round(xi_U / TWO_PI)
    else:
    	xi_Lpbc = xi_L 
    	xi_Upbc = xi_U
    print(f"\n RESULTS:")
    print("Number of Hills Read =", n)
    print("Lambda =", lam)
    print("xi_lower= ", xi_Lpbc)
    print("xi_upper= ", xi_Upbc)

if __name__ == "__main__":
    compute_lambda()
