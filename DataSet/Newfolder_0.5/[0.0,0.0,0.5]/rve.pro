timeStep = 5.0e-5;

log =
{
  pattern = "*.info";
  file    = "-$(CASE_NAME).log";
};

control	= 
{
  fgMode	= false;
  runWhile	= "i<50";
};

userinput =
{
  modules = [ "mesh", "pbc", "hardening" ];

  hardening =
  {
    type = "Input";
    file = "hmelro3.data";
  };

  mesh = 
  {
    type = "GmshInput";
    file = "rve.msh";
    doElemGroups = true;
  };

  pbc = 
  {
    type = "PBCGroupInput";
  };
};


model =
{
  type        = "Matrix";
  matrix.type = "Sparse";

  model       =
  {
    type   = "Multi";
    models = [ "matrix", "fibers", "fixed", "bc", "pbc", "lodileft", "lodiright", "lodibot", "loditop" ];

    loditop =
    {
      type = "LoadDisp";
      group = "ymax";
    };

    lodibot =
    {
      type = "LoadDisp";
      group = "ymin";
    };

    lodileft =
    {
      type = "LoadDisp";
      group = "xmin";
    };

    lodiright =
    {
      type = "LoadDisp";
      group = "xmax";
    };

    pbc =
    {
      type = "PeriodicBC";
    };

    fixed =
    {
      type = "BC";
      mode = "disp";

      nodeGroups = [ "cornerx" ];
      dofs   = [ "dy" ];
      unitVec = [ 0.0 ];

      shape = "t";
      step = timeStep;
    };

    bc =
    {
      type = "BC";
      mode = "disp";

      include "bcs";

      shape = "t";
      step = timeStep;
    };

    matrix =
    {
      type     = "Stress";
      elements = "gmsh1";

      material =
      {
	type   = "Melro";
	rank   = 2;
	anmodel = "PLANE_STRAIN";

	E = 3760.;
	nu = 0.3;
	poissonP = 0.39;
	sigmaT = "st(x)";
	sigmaC = "sc(x)";
	rmTolerance = 1e-10;
	rmMaxIter   = 1000;
      };
      
      shape.type = "Triangle3";
      shape.intScheme = "Gauss1";
    };

    fibers =
    {
      type = "Stress";
      elements = "gmsh0";

      material =
      {
	type = "Isotropic";
	rank = 2;
	anmodel = "PLANE_STRAIN";

	E  = 74000.;
	nu = 0.2;
      };

      shape.type = "Triangle3";
      shape.intScheme = "Gauss1";
    };
  };
};

usermodules = 
{
  modules = [ "stepper", "sample", "paraview" ];//, "view" ];//, "graph", "view" ];

  sample =
  {
    type = "Sample";
    file = "lodi.dat";
    sampleWhen = "accepted";
    dataSets = [ "i", 
                 "model.model.lodiright.disp[0] - model.model.lodileft.disp[0]", 
		 "model.model.lodiright.load[0]",
                 "model.model.loditop.disp[1] - model.model.lodibot.disp[1]", 
		 "model.model.loditop.load[1]",
                 "model.model.loditop.disp[0] - model.model.lodibot.disp[0]", 
		 "model.model.loditop.load[0]" ];
  };

  paraview =
  {
    type = "ParaView";
    groups = [ "gmsh1" ];
    
    gmsh1 =
    {
      shape = "Triangle3";
      disps = [ "dx", "dy" ];

      node_data = [ "nodalStress", "nodalHistory" ];
    };
  };

  stepper = 
  {
    type = "AdaptiveStep";

    optIter = 5;
    reduction = 0.8;
    minIncr = timeStep;
    maxIncr = timeStep;
    startIncr = timeStep;

    solver = 
    {
      type = "Nonlin";
      precision = 1.e-4;
      solver.type = "SkylineLU";
      maxIter = 20;
    };
  };

  view = 
  {
    type = "FemView";

    updateWhen = true;
    dataSets = [ "hist" ];

    hist.type = "Table";
    hist.table= "nodes/nodalHistory";

    mesh = 
    {
      plugins = "colors";

      colors = 
      {
        type = "MeshColorView";
        data = "hist[epspeq]";
      };
    };
  };
};
