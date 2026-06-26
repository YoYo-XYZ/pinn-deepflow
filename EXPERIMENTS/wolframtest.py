from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl
from wolframclient.language import Global as gl

# Start Wolfram session
session = WolframLanguageSession()

# Define viscosity
nu = 0.01 / 3.141592653589793

# Define PDE in Wolfram Language
pde = wl.Equal(
    wl.Plus(
        wl.D(gl.u(gl.x, gl.t), gl.t),
        wl.Times(gl.u(gl.x, gl.t), wl.D(gl.u(gl.x, gl.t), gl.x))
    ),
    wl.Times(nu, wl.D(gl.u(gl.x, gl.t), gl.x, gl.x))
)

# Initial and boundary conditions
ics = wl.Equal(gl.u(gl.x, 0), wl.Times(-1, wl.Sin(wl.Times(wl.Pi, gl.x))))

bcs = [
    wl.Equal(gl.u(-1, gl.t), gl.u(1, gl.t)),
    wl.Equal(
        wl.ReplaceAll(wl.D(gl.u(gl.x, gl.t), gl.x), wl.Rule(gl.x, -1)),
        wl.ReplaceAll(wl.D(gl.u(gl.x, gl.t), gl.x), wl.Rule(gl.x, 1))
    )
]

# Solve PDE
solution = session.evaluate(
    wl.NDSolve(
        [pde, ics] + bcs,
        gl.u,
        (gl.x, -1, 1),
        (gl.t, 0, 1)
    )
)

print("Solution obtained from Wolfram Engine.")
