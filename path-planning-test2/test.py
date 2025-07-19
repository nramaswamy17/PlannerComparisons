import python_motion_planning as pmp
env     = pmp.Grid(51, 31)
planner = pmp.DStarLite(start=(5,5), goal=(48,28), env=env)
cost, path, expanded = planner.plan()
planner.plot.animation(path, "D* Lite", cost, expanded)
