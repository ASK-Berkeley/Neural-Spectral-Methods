(* Load parameters from numpy file *)
Source = Last[ExternalEvaluate["Python", {
  "import numpy as np",
  "np.load(\"s.dirichlet.npy\")"
}]];

(* Qurey grid points: uniformly spaced *)
X = CoordinateBoundsArray[{{0, 255}, {0, 255}}] / 255;        

PDE = -Div[Grad[u[x, y], {x, y}], {x, y}] == s;
BCs = u[0, y] == u[1, y] == u[x, 0] == u[x, 1] == 0;

U = Table[

    {nx, ny, one} = Dimensions[S];

    cx = ChebyshevT[#, x] & /@ Range[0, nx-1] /. x -> 2x - 1;
    cy = ChebyshevT[#, y] & /@ Range[0, ny-1] /. y -> 2y - 1;
    s = Sum[S[[i, j, 1]] cx[[i]] cy[[j]], {i, nx}, {j, ny}];

    (* Solve PDE & measure cost *)
    { Cost, Evaluation } = Timing[
        Solution = NDSolveValue[{PDE, BCs}, u, {x, 0, 1}, {y, 0, 1}];
        ArrayReshape[Solution @@@ Flatten[X, 1], {255, 255, 1}]
    ];

    Print[Cost];
    Evaluation

, { S, Normal[Source] }];

(* Store answer to `.npy` *)
ExternalEvaluate["Python", {
    "import numpy as np",
    "np.save(\"u.dirichlet.npy\", <*NumericArray[U, \"Real32\"]*>)",
    "print('DONE')"
}];
