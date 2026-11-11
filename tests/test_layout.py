"""Run the layout checker on the existing sketches (expect case-2 overlaps to be caught)."""
from cfdlib import plots
from cases.plane_poiseuille import sketch as s1
from cases.lid_driven_cavity_re100 import sketch as s2
for name, fn in (("poiseuille", lambda: s1.sketch()), ("cavity", lambda: s2.sketch((0.6172, 0.7344)))):
    try:
        fn(); print(name, "OK")
    except plots.FigureLayoutError as e:
        print(name, e)
