from solve import lbm_solver
from GUI import initial

def main():
    lbm = lbm_solver()
    gui1 = initial(lbm)
    gui1.UI(lbm)


if __name__ == '__main__':
    main()
