#!/usr/bin/env python3
import os, sys

# Ensure ISSM paths are correctly set
issm_dir = os.getenv('ISSM_DIR')
if not issm_dir:
    print("Error: ISSM_DIR environment variable is not set.")
    sys.exit(1)


sys.path.insert(0, os.path.join(issm_dir, 'src', 'm', 'boundaryconditions'))
sys.path.insert(0, os.path.join(issm_dir, 'src', 'm', 'classes'))
sys.path.insert(1, os.path.join(issm_dir, 'src', 'm', 'classes', 'clusters'))
sys.path.append(os.path.join(issm_dir, 'src', 'm', 'contrib', 'jkjhew'))

sys.path.append(issm_dir + '/bin')
sys.path.append(issm_dir + '/lib')

import numpy as np
from triangle import triangle
from model import *
from netCDF4 import Dataset
from InterpFromGridToMesh import InterpFromGridToMesh
from bamg import bamg
from xy2ll import xy2ll
from plotmodel import plotmodel
from export_netCDF import export_netCDF
from loadmodel import loadmodel
from setmask import setmask
from parameterize import parameterize
from setflowequation import setflowequation
from socket import gethostname
from solve import solve
from ll2xy import ll2xy
from BamgTriangulate import BamgTriangulate
from InterpFromMeshToMesh2d import InterpFromMeshToMesh2d
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from gadi_spack import gadi
from organizer import organizer
from toolkits import toolkits
from bcgslbjacobioptions import bcgslbjacobioptions
from SetMOLHOBC import SetMOLHOBC

# 1) Define the steps and model number
steps = [11]
modelnum = 1

# 2) Choose model name from model number
if modelnum == 1:
    modelname = '1km_viscous'
elif modelnum == 2:
    modelname = '2km_viscous'
elif modelnum == 3:
    modelname = '1km_coulomb'
elif modelnum == 4:
    modelname = '2km_coulomb'
elif modelnum == 5:
    modelname = '500m_viscous'
elif modelnum == 6:
    modelname = '500m_coulomb'
elif modelnum == 7:
    modelname = '200m_viscous'
elif modelnum == 8:
    modelname = '200m_coulomb'
else:
    raise ValueError('Invalid modelnum!')

# Hard-coded parameters
clustername = 'gadi'
# loadonly = 0
# interactive = 0
printflag = True



if clustername.lower() == 'gadi':

    cluster = gadi(
        'name',         'gadi.nci.org.au',
        'login',        'jh7060',
        'srcpath',      '/g/data/au88/jh7060/ISSM',
        'project',      'au88',
        'numnodes',     1,
        'cpuspernode',  32,
        'time',         48*60,  # minutes
        'codepath',     '/g/data/au88/jh7060/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/issm-4.24-v52nx3pfx7lpqwfldqlpspflk34wz756/bin',
        'executionpath','/scratch/au88/jh7060/issm_runs'
    )
    loadonly = 0
    lock     = 1
else:
    # --- generic single-node/local fallback ---
    cluster        = generic()            # <-- no args here
    cluster.name   = gethostname()        # label for logs
    cluster.np     = 32              # number of MPI processes
    cluster.mem    = 64              # GB – change to suit your machine
    cluster.time   = 60                  # wall-clock minutes
    cluster.queue  = 'local'              # or the default queue on your scheduler
    lock     = 1
    loadonly = 1

# Create an organizer
# - The “organizer” is the same logic as the MATLAB version.
# - Note that in Python, you typically use Organizer(...) from issm.
org = organizer(
     'repository',   './Models_'     + modelname,
     'prefix',       'mismip_'       + modelname + '_',
     'steps',        steps,
     'trunkprefix',  '34;47;2'
)
# ------------------------------------------------------------------------------
#  STEP: Mesh_generation
# ------------------------------------------------------------------------------
if org.perform('Mesh_generation'):
    model = model()  # Start a blank model
    if modelnum == 1 or modelnum == 3:
        md = bamg(model,
          'domain',       './Domain.exp',
          'hmax',         1000,
          'splitcorners', 1)

        print(md.mesh.x.max() -  md.mesh.x.min())
    elif modelnum == 2 or modelnum == 4:
        md = bamg(model, domain='./Domain.exp', hmax=2000, splitcorners=1)
    elif modelnum == 5 or modelnum == 6:
        md = bamg(model, domain='./Domain.exp', hmax=500, splitcorners=1)
    elif modelnum == 7 or modelnum == 8:
        md = bamg(model, domain='./Domain.exp', hmax=200, splitcorners=1)
    else:
        raise RuntimeError("Model not supported yet")

    md.miscellaneous.name = 'MISMIP_' + modelname
    org.savemodel(md)

# ------------------------------------------------------------------------------
#  STEP: Parameterization
# ------------------------------------------------------------------------------
if org.perform('Parameterization'):
    md = org.loadmodel('Mesh_generation')
    md = setmask(md, '', '')
    md = parameterize(md, './Mismip.py')
    org.savemodel(md)

# ------------------------------------------------------------------------------
#  STEP: Transient_Steadystate
# ------------------------------------------------------------------------------
if org.perform('Transient_Steadystate'):
    md = org.loadmodel('Parameterization')
    md = setflowequation(md, 'SSA', 'all')

    # Coulomb friction for certain models
    if (modelnum == 3) or (modelnum == 4) or (modelnum == 6):
        md.friction = frictioncoulomb()
        md.friction.coefficient = math.sqrt(3.160e6) * np.ones(md.mesh.numberofvertices)
        md.friction.coefficientcoulomb = math.sqrt(0.5) * np.ones(md.mesh.numberofvertices)
        md.friction.p = 3 * np.ones(md.mesh.numberofelements)
        md.friction.q = np.zeros(md.mesh.numberofelements)

    md.timestepping.time_step = 1
    md.timestepping.final_time = 200000
    md.settings.output_frequency = 2000
    md.settings.checkpoint_frequency = 2000
    md.stressbalance.maxiter = 30
    md.stressbalance.abstol = float('nan')
    md.stressbalance.restol = 1
    md.verbose = verbose('solution', True, 'module', True, 'convergence', True)
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_Tss1'
    md.settings.solver_residue_threshold = float('nan')

    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype,'loadonly', loadonly, 'lock', lock, 'runtimename', False)
    export_netCDF(md, "./mismip_1km_viscous_Transient_steadystate.nc")
    org.savemodel(md)

# ------------------------------------------------------------------------------
#  STEP: Transient_Steadystate_remesh (example of specialized re-meshing)
# ------------------------------------------------------------------------------
if org.perform('Transient_Steadystate_remesh'):
    md = org.loadmodel('Parameterization')
    # Example: load from another directory
    # The code below tries to load an existing model from the 2km_viscous folder.
    # Adapt to your local file paths if needed.
    md2 = loadmodel('Models_2km_viscous/mismip_2km_viscous_Transient_steadystate2.nc')
    # Overwrite the final solution’s mesh
    md2.results.TransientSolution[-1].MeshX = md.mesh.x
    md2.results.TransientSolution[-1].MeshY = md.mesh.y
    md2.results.TransientSolution[-1].MeshElements = md.mesh.elements

    md = remesh(md2, './Mismip.py')
    md = setflowequation(md, 'SSA', 'all')
    if (modelnum == 3) or (modelnum == 4) or (modelnum == 6):
        md.friction = frictioncoulomb()
        md.friction.coefficient = math.sqrt(3.160e6) * np.ones(md.mesh.numberofvertices)
        md.friction.coefficientcoulomb = math.sqrt(0.5) * np.ones(md.mesh.numberofvertices)
        md.friction.p = 3 * np.ones(md.mesh.numberofelements)
        md.friction.q = np.zeros(md.mesh.numberofelements)


    md.timestepping.time_step = 1
    md.timestepping.final_time = 200000
    md.settings.output_frequency = 5000
    md.settings.checkpoint_frequency = 5000
    md.stressbalance.maxiter = 10
    md.stressbalance.abstol = float('nan')
    md.stressbalance.restol = 1
    md.verbose = verbose('convergence', False, 'solution', False)
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_Tssr'
    md.settings.solver_residue_threshold = float('nan')
    md.settings.waitonlock = 0

    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype)
    org.savemodel(md)
 

# ------------------------------------------------------------------------------
#  STEP: Transient_steadystate2
# ------------------------------------------------------------------------------
if org.perform('Transient_steadystate2'):
    md = org.loadmodel('Transient_Steadystate')

    md = setflowequation(md, 'SSA', 'all')
    # Re-initialize from previous solution
    md.initialization.vx = md.results.TransientSolution[-1].Vx
    md.initialization.vy = md.results.TransientSolution[-1].Vy
    md.initialization.vel = md.results.TransientSolution[-1].Vel
    md.geometry.thickness = md.results.TransientSolution[-1].Thickness
    md.geometry.base = md.results.TransientSolution[-1].Base
    md.geometry.surface = md.results.TransientSolution[-1].Surface
    md.mask.ocean_levelset = md.results.TransientSolution[-1].MaskOceanLevelset

    md.timestepping.time_step = 1
    md.timestepping.final_time = 10000
    md.settings.output_frequency = 1000
    md.settings.checkpoint_frequency = 5000
    md.stressbalance.maxiter = 10
    md.stressbalance.abstol = float('nan')
    md.stressbalance.restol = 1
    md.verbose = verbose('solution', True, 'module', True, 'convergence', True)
    md.cluster = cluster
    md.settings.waitonlock = 1
    md.miscellaneous.name = 'MISMIP_' + modelname + '_Tss2'

    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype,'loadonly', loadonly, 'lock', lock)#, 'runtimename', False)
    #export_netCDF(md, "./mismip_1km_viscous_Transient_steadystate_2.nc")

# ------------------------------------------------------------------------------
#  STEP: Transient_steadystate3
# ------------------------------------------------------------------------------
if org.perform('Transient_steadystate3'):
    md = org.loadmodel('Transient_steadystate2')
    md = setflowequation(md, 'SSA', 'all')
    # Re-initialize from previous solution
    md.initialization.vx = md.results.TransientSolution[-1].Vx
    md.initialization.vy = md.results.TransientSolution[-1].Vy
    md.initialization.vel = md.results.TransientSolution[-1].Vel
    md.geometry.thickness = md.results.TransientSolution[-1].Thickness
    md.geometry.base = md.results.TransientSolution[-1].Base
    md.geometry.surface = md.results.TransientSolution[-1].Surface
    md.mask.ocean_levelset = md.results.TransientSolution[-1].MaskOceanLevelset

    md.timestepping.time_step = 1
    md.timestepping.final_time = 200000
    md.settings.output_frequency = 6000
    md.stressbalance.maxiter = 10
    md.stressbalance.abstol = float('nan')
    md.stressbalance.restol = 1
    md.verbose = verbose('solution', True, 'module', True, 'convergence', True)
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_Tss3'
    md.settings.solver_residue_threshold = float('nan')
    md.settings.waitonlock = 0
    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype,'loadonly', loadonly, 'lock', lock)
    org.savemodel(md)

# ------------------------------------------------------------------------------
#  STEP: Transient_steadystate4
# ------------------------------------------------------------------------------
if org.perform('Transient_steadystate4'):
    md = org.loadmodel('Transient_steadystate3')
    md = setflowequation(md, 'SSA', 'all')
    # Re-initialize from previous solution
    md.initialization.vx = md.results.TransientSolution[-1].Vx
    md.initialization.vy = md.results.TransientSolution[-1].Vy
    md.initialization.vel = md.results.TransientSolution[-1].Vel
    md.geometry.thickness = md.results.TransientSolution[-1].Thickness
    md.geometry.base = md.results.TransientSolution[-1].Base
    md.geometry.surface = md.results.TransientSolution[-1].Surface
    md.mask.ocean_levelset = md.results.TransientSolution[-1].MaskOceanLevelset

    md.timestepping.time_step = 1
    md.timestepping.final_time = 200000
    md.settings.output_frequency = 6000
    md.stressbalance.maxiter = 10
    md.stressbalance.abstol = float('nan')
    md.stressbalance.restol = 1
    md.verbose = verbose('convergence', False, 'solution', True)
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_Tss4'

    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype)
    org.savemodel(md)

# ------------------------------------------------------------------------------
#  STEP: Transient_steadystate5
# ------------------------------------------------------------------------------
if org.perform('Transient_steadystate5'):
    md = org.loadmodel('Transient_steadystate4')
    md = setflowequation(md, 'SSA', 'all')
    # Re-initialize from previous solution
    md.initialization.vx = md.results.TransientSolution[-1].Vx
    md.initialization.vy = md.results.TransientSolution[-1].Vy
    md.initialization.vel = md.results.TransientSolution[-1].Vel
    md.geometry.thickness = md.results.TransientSolution[-1].Thickness
    md.geometry.base = md.results.TransientSolution[-1].Base
    md.geometry.surface = md.results.TransientSolution[-1].Surface
    md.mask.ocean_levelset = md.results.TransientSolution[-1].MaskOceanLevelset

    md.timestepping.time_step = 1
    md.timestepping.final_time = 200000
    md.settings.output_frequency = 6000
    md.stressbalance.maxiter = 10
    md.stressbalance.abstol = float('nan')
    md.stressbalance.restol = 1
    md.verbose = verbose('convergence', False, 'solution', True)
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_Tss5'
    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype)
    org.savemodel(md)

# ------------------------------------------------------------------------------
#  STEP: Transient_extrude
# ------------------------------------------------------------------------------
if org.perform('Transient_extrude'):
    md = org.loadmodel('Transient_steadystate3')
    # Re-initialize from previous solution
    md.initialization.vx = md.results.TransientSolution[-1].Vx
    md.initialization.vy = md.results.TransientSolution[-1].Vy
    md.initialization.vel = md.results.TransientSolution[-1].Vel
    md.geometry.thickness = md.results.TransientSolution[-1].Thickness
    md.geometry.base = md.results.TransientSolution[-1].Base
    md.geometry.surface = md.results.TransientSolution[-1].Surface
    md.mask.ocean_levelset = md.results.TransientSolution[-1].MaskOceanLevelset

    md = md.extrude( 10, 1.1)
    md = setflowequation(md, 'HO', 'all')
    md.transient.isthermal = 0
    md.transient.issmb = 0
    md.initialization.temperature[:] = 273.0

    org.savemodel(md)

# ------------------------------------------------------------------------------
#  STEP: GlenSSA
# ------------------------------------------------------------------------------
if org.perform('GlenSSA'):
    if modelnum == 5 or modelnum == 6:
        md = org.loadmodel('Transient_Steadystate') #200k years relaxation
    else:
        md = org.loadmodel('Transient_extrude') #600k years relaxation

    md.transient.requested_outputs = [
        'default','IceVolume','IceVolumeAboveFloatation','GroundedArea',
        'StrainRatexx','StrainRatexy','StrainRateyy','StrainRateeffective',
        'MaskOceanLevelset','IceMaskNodeActivation','MaskIceLevelset'
    ]
    md = md.collapse()  # collapse 3D model to 2D if extruded
    md = setflowequation(md, 'SSA', 'all')

    md.timestepping.time_step = 1.0/12.0
    md.timestepping.final_time = 1000
    md.settings.output_frequency = 600
    md.stressbalance.maxiter = 10
    md.stressbalance.restol = 1
    md.stressbalance.reltol = 0.001
    md.stressbalance.abstol = float('nan')
    md.verbose.convergence = False
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_GSSA'

    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype)
    org.savemodel(md)

# ------------------------------------------------------------------------------
#  STEP: GlenMOLHO
# ------------------------------------------------------------------------------
if org.perform('GlenMOLHO'):
    if modelnum == 5 or modelnum == 6:
        md = org.loadmodel('Transient_Steadystate')
    else:
        md = org.loadmodel('Transient_extrude')

    md.transient.requested_outputs = [
        'default','IceVolume','IceVolumeAboveFloatation','GroundedArea',
        'VxShear','VyShear','VxBase','VyBase','VxSurface','VySurface',
        'VxAverage','VyAverage','StrainRatexx','StrainRatexy','StrainRateyy','StrainRateeffective',
        'MaskOceanLevelset','IceMaskNodeActivation','MaskIceLevelset'
    ]
    md = md.collapse()
    md = setflowequation(md, 'MOLHO', 'all')

    # Example solver settings
    md.stressbalance.maxiter = 10
    md.stressbalance.restol = 1
    md.stressbalance.reltol = 0.001
    md.stressbalance.abstol = float('nan')
    # If you need special solver settings for small meshes:
   # md.toolkits = toolkits()
    md.toolkits = toolkits.addoptions(md.toolkits,'StressbalanceAnalysis',bcgslbjacobioptions())

    md = SetMOLHOBC(md)  # Shear boundary conditions for MOLHO?

    md.timestepping.time_step = 1.0/12.0
    md.timestepping.final_time = 1000
    md.settings.output_frequency = 600
    md.verbose.convergence = False
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_GMOLHO'

    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype)
    org.savemodel(md)

# ------------------------------------------------------------------------------
#  STEP: GlenHO
# ------------------------------------------------------------------------------
if org.perform('GlenHO'):
    if modelnum == 5 or modelnum == 6:
        md = org.loadmodel('Transient_Steadystate')
    else:
        md = org.loadmodel('Transient_extrude')

    md = setflowequation(md, 'HO', 'all')
    md.transient.requested_outputs = [
        'default','IceVolume','IceVolumeAboveFloatation','GroundedArea',
        'StrainRatexx','StrainRatexy','StrainRateyy','StrainRatexz','StrainRateyz','StrainRatezz',
        'StrainRateeffective',
        'MaskOceanLevelset','IceMaskNodeActivation','MaskIceLevelset'
    ]
    md.timestepping.time_step = 1.0/12.0
    md.timestepping.final_time = 1000
    md.settings.output_frequency = 600
    #md.toolkits = md.addoptions(md.toolkits, 'StressbalanceAnalysis', bcgslbjacobioptions())
    md.stressbalance.maxiter = 10
    md.stressbalance.restol = 1
    md.stressbalance.reltol = 0.001
    md.stressbalance.abstol = float('nan')
    md.verbose.convergence = False
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_GHO'

    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype)
    org.savemodel(md)

# ------------------------------------------------------------------------------
#  STEP: GlenFS
# ------------------------------------------------------------------------------
if org.perform('GlenFS'):
    # If using a 2D mesh (e.g. 500m case), use the steady state model
    if modelnum == 5 or modelnum == 6:
        md = org.loadmodel('Transient_Steadystate')
    else:
        # or extruded model
        md = org.loadmodel('Transient_steadystate3')

        # or load from GlenHO to reuse geometry
        # md = org.loadmodel('GlenHO')
    md.initialization.vx = md.results.TransientSolution[-1].Vx
    md.initialization.vy = md.results.TransientSolution[-1].Vy
    md.initialization.vel = md.results.TransientSolution[-1].Vel
    md.geometry.thickness = md.results.TransientSolution[-1].Thickness
    md.geometry.base = md.results.TransientSolution[-1].Base
    md.geometry.surface = md.results.TransientSolution[-1].Surface
    md.mask.ocean_levelset = md.results.TransientSolution[-1].MaskOceanLevelset

    md = md.extrude( 10, 1.1)
    md = setflowequation(md, 'HO', 'all')
    md.transient.isthermal = 0
    md.transient.issmb = 0
    #md.transient.ismasstransport = 0
    md.initialization.temperature[:] = 273.0
    
    md = setflowequation(md, 'FS', 'all')
    md.stressbalance.shelf_dampening = 1
    md.masstransport.isfreesurface = 1
    md.transient.isgroundingline = 1  # or 1, depending on your test
    md.groundingline.migration='Contact' # or 'SoftMigration'
    md.timestepping.time_step = 0.00001
    md.timestepping.final_time = 0.0001
    md.settings.output_frequency = 1
   # md = md.SetInput(md, 'Bed', md.geometry.bed)
    md.toolkits = toolkits.addoptions(md.toolkits, 'StressbalanceAnalysis', bcgslbjacobioptions())
    md.flowequation.fe_FS = 'TaylorHood'
    md.stressbalance.maxiter = 20
    md.stressbalance.restol = 0.5
    md.stressbalance.reltol = 0.001
    # If you know surface and thickness:
    #md.geometry.bed = md.geometry.surface - md.geometry.thickness

    #md.geometry.surface = InterpFromGridToMesh(x, y, usrf, md.mesh.x, md.mesh.y, 0);
    #md.geometry.bed     = InterpFromGridToMesh(x, y, topg, md.mesh.x, md.mesh.y, 0);


    # Or if you already interpolated base (from NetCDF):
#; % Sometimes 'base' is used internally

    md.stressbalance.abstol = float('nan')
    md.verbose.convergence = False
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_GFS'
    print(md.mesh)

    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype,'loadonly', loadonly, 'lock', lock, 'runtimename', False)
    #org.savemodel(md)

# ------------------------------------------------------------------------------
#  STEP: GlenESSA (enhanced SSA with a factor E=5)
# ------------------------------------------------------------------------------
if org.perform('GlenESSA'):
    if modelnum == 5 or modelnum == 6:
        md = org.loadmodel('Transient_Steadystate')
    else:
        md = org.loadmodel('Transient_extrude')

    md.transient.requested_outputs = [
        'default','IceVolume','IceVolumeAboveFloatation','GroundedArea',
        'StrainRatexx','StrainRatexy','StrainRateyy','StrainRateeffective',
        'MaskOceanLevelset','IceMaskNodeActivation','MaskIceLevelset'
    ]
    md = collapse(md)
    md = setflowequation(md, 'SSA', 'all')
    md.materials = matenhancedice(md.materials)
    md.materials.rheology_E = 5.0 * np.ones(md.mesh.numberofvertices)

    md.timestepping.time_step = 1.0/12.0
    md.timestepping.final_time = 1000
    md.settings.output_frequency = 600
    md.stressbalance.maxiter = 10
    md.stressbalance.restol = 1
    md.stressbalance.reltol = 0.001
    md.stressbalance.abstol = float('nan')
    md.verbose.convergence = False
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_ESSA'

    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype)
    org.savemodel(md)

# ------------------------------------------------------------------------------
#  STEP: GlenEMOLHO (enhanced MOLHO)
# ------------------------------------------------------------------------------
if org.perform('GlenEMOLHO'):
    if modelnum == 5 or modelnum == 6:
        md = org.loadmodel('Transient_Steadystate')
    else:
        md = org.loadmodel('Transient_extrude')

    md.transient.requested_outputs = [
        'default','IceVolume','IceVolumeAboveFloatation','GroundedArea',
        'VxShear','VyShear','VxBase','VyBase','VxSurface','VySurface',
        'VxAverage','VyAverage','StrainRatexx','StrainRatexy','StrainRateyy','StrainRateeffective',
        'MaskOceanLevelset','IceMaskNodeActivation','MaskIceLevelset'
    ]
    md = collapse(md)
    md = setflowequation(md, 'MOLHO', 'all')
    md.materials = matenhancedice(md.materials)
    md.materials.rheology_E[:] = 5.0

    md = SetMOLHOBC(md)

    md.stressbalance.maxiter = 10
    md.stressbalance.restol = 1
    md.stressbalance.reltol = 0.001
    md.stressbalance.abstol = float('nan')

    md.timestepping.time_step = 1.0/12.0
    md.timestepping.final_time = 1000
    md.settings.output_frequency = 600
    md.verbose.convergence = False
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_GEMOLHO'

    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype)
    org.savemodel(md)
# ------------------------------------------------------------------------------
#  STEP: GlenEHO (enhanced HO)
# ------------------------------------------------------------------------------
if org.perform('GlenEHO'):
    if modelnum == 5 or modelnum == 6:
        md = org.loadmodel('Transient_Steadystate')
    else:
        md = org.loadmodel('Transient_extrude')

    md = setflowequation(md, 'HO', 'all')
    md.materials = matenhancedice(md.materials)
    md.materials.rheology_E = 5.0 * np.ones(md.mesh.numberofvertices)
    md.transient.requested_outputs = [
        'default','IceVolume','IceVolumeAboveFloatation','GroundedArea',
        'StrainRatexx','StrainRatexy','StrainRateyy','StrainRatexz','StrainRateyz','StrainRatezz',
        'StrainRateeffective',
        'MaskOceanLevelset','IceMaskNodeActivation','MaskIceLevelset'
    ]
    md.timestepping.time_step = 1.0/12.0
    md.timestepping.final_time = 1000
    md.settings.output_frequency = 600
    md.toolkits = md.addoptions(md.toolkits, 'StressbalanceAnalysis', bcgslbjacobioptions())
    md.stressbalance.maxiter = 10
    md.stressbalance.restol = 1
    md.stressbalance.reltol = 0.001
    md.stressbalance.abstol = float('nan')
    md.verbose.convergence = True
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_GEHO'

    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype)
    org.savemodel(md)

# ------------------------------------------------------------------------------
#  STEP: GlenEFS (enhanced FS)
# ------------------------------------------------------------------------------
if org.perform('GlenEFS'):
    if modelnum == 5 or modelnum == 6:
        md = org.loadmodel('Transient_Steadystate')
    else:
        md = org.loadmodel('Transient_extrude')

    md = setflowequation(md, 'FS', 'all')
    md.materials = matenhancedice(md.materials)
    md.materials.rheology_E[:] = 5.0
    md.stressbalance.shelf_dampening = 1
    md.masstransport.isfreesurface = 1
    md.groundingline.migration = 'Contact'

    # Example short run
    md.timestepping.time_step = 0.001
    md.timestepping.final_time = 0.002
    md.settings.output_frequency = 1

    md.stressbalance.maxiter = 10
    md.stressbalance.restol = 1
    md.stressbalance.reltol = 0.001
    md.stressbalance.abstol = float('nan')
    md.verbose.convergence = True
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_GEFS'

    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype)
    org.savemodel(md)

if org.perform("ESTARSSA"):
    if modelnum == 5 or modelnum == 6:
        md = org.loadmodel('Transient_Steadystate')
    else:
        md = org.loadmodel('Transient_extrude')
    md = setflowequation(md, 'SSA', 'all')
    md.materials = matenhancedice(md.materials)
    md.materials.rheology_Es = 5.0 * np.ones(md.mesh.numberofvertices)
    md.materials.rheology_Ec= 3.0/8.0 * md,materials.rheology_Es
    md.transient.requested_outputs = ['default','IceVolume','IceVolumeAboveFloatation','GroundedArea',
		'StrainRatexx','StrainRatexy','StrainRateyy','StrainRateeffective','LambdaS','Epsprime',
		'MaskOceanLevelset','IceMaskNodeActivation','MaskIceLevelset']
    
    md.timestepping.time_step = 1.0/12.0
    md.timestepping.final_time = 1000
    md.settings.output_frequency = 600
    md.stressbalance.maxiter = 10
    md.stressbalance.restol = 1
    md.stressbalance.reltol = 0.001
    md.stressbalance.abstol = float('nan')
    md.verbose.convergence = False
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_ESTARSSA'
    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype)
    org.savemodel(md)

if org.perform("ESTARMOLHO"):

    if modelnum == 5 or modelnum == 6:
        md = org.loadmodel('Transient_Steadystate')
    else:
        md = org.loadmodel('Transient_extrude')
    
    md.transient.requested_outputs = ['default','IceVolume','IceVolumeAboveFloatation','GroundedArea',
		'VxShear','VyShear','VxBase','VyBase','VxSurface','VySurface','VxAverage','VyAverage',
		'StrainRatexx','StrainRatexy','StrainRateyy','StrainRateeffective','LambdaS','Epsprime',
		'MaskOceanLevelset','IceMaskNodeActivation','MaskIceLevelset']
    md = collapse(md)
    md = setflowequation(md, 'MOLHO', 'all')
    md.materials = matestar(md.materials)
    md.materials.rheology_Es = 5.0 * np.ones(md.mesh.numberofvertices)
    md.materials.rheology_Ec= 3.0/8.0 * md,materials.rheology_Es

    md.stressbalance.maxiter = 10
    md.stressbalance.restol = 1
    md.stressbalance.reltol = 0.001
    md.stressbalance.abstol = float('nan')
    if res <= 1000:
        md.toolkits = toolkits()
        md.toolkits = addoptions(md.toolkits, 'StressbalanceAnalysis', bcgslbjacobioptions())

        md.settings.solver_residue_threshold = np.nan  # 11/05/2019
        md.stressbalance.maxiter = 50                 # 10/24/2019
        md.stressbalance.restol = 1e-4                # 12/17/2019 original
        md.stressbalance.reltol = np.nan              # 11/05/2019
        md.stressbalance.abstol = np.nan              # 11/05/2019

    md.timestepping.time_step = 1.0/12.0
    md.timestepping.final_time = 1000
    md.settings.output_frequency = 600
    md.verbose.convergence = False
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_ESTARMOLHO'
    
    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype)
    org.savemodel(md)

# ------------------------------------------------------------------------------
if org.perform("ESTARHO"):
    if modelnum == 5 or modelnum == 6:
        md = org.loadmodel('Transient_Steadystate')
    else:
        md = org.loadmodel('Transient_extrude')
    md = setflowequation(md, 'HO', 'all')
    md.materials = matestar(md.materials)
    md.transient.requested_outputs = ['default','IceVolume','IceVolumeAboveFloatation','GroundedArea',
		'StrainRatexx','StrainRatexy','StrainRateyy','StrainRatexz','StrainRateyz','StrainRatezz',
		'StrainRateeffective','LambdaS',
		'MaskOceanLevelset','IceMaskNodeActivation','MaskIceLevelset']
    md.materials.rheology_Es = 5.0 * np.ones(md.mesh.numberofvertices)
    md.materials.rheology_Ec= 3.0/8.0 * md,materials.rheology_Es

    md.timestepping.time_step = 1.0/12.0
    md.timestepping.final_time = 1000
    md.settings.output_frequency = 600
    md.toolkits = addoptions(md.toolkits, 'StressbalanceAnalysis', bcgslbjacobioptions())
    md.stressbalance.maxiter = 30
    md.stressbalance.restol = 1
    md.stressbalance.reltol = 0.001
    md.stressbalance.abstol = float('nan')
    md.verbose.convergence = False
    md.cluster = cluster
    md.miscellaneous.name = 'MISMIP_' + modelname + '_EFS'
    
    solutiontype = 'tr'    # 'tr' = transient, 'sb' = stress‑balance, etc.
    md = solve(md, solutiontype)
    org.savemodel(md)

# ------------------------------------------------------------------------------
#  STEP: (Example) Analysis / Plotting
# ------------------------------------------------------------------------------
if org.perform('analyse'):
    mdgs = org.loadmodel('GlenSSA')
    mdgm = org.loadmodel('GlenMOLHO')
    mdgh = org.loadmodel('GlenHO')
    mdes = org.loadmodel('ESTARSSA')  # e.g., or GlenESSA
    mdeh = org.loadmodel('ESTARHO')   # e.g., or GlenEHO

    mdgsV = mdgs.results.TransientSolution[-1].Vel
    mdgmV = mdgm.results.TransientSolution[-1].Vel
    mdghV = mdgh.results.TransientSolution[-1].Vel
    mdesV = mdes.results.TransientSolution[-1].Vel
    mdehV = mdeh.results.TransientSolution[-1].Vel

    plotmodel(mdgs,'data',mdgsV,'data',mdgmV,'data',mdghV(mdeh.mesh.vertexonsurface==1),
    'data',mdesV,'data',mdehV(mdeh.mesh.vertexonsurface==1),'nlines',5,'ncols',1,'caxis#all',[1,1000])
    plt.show()
    # Plot with Python version of plotmodel (if available),
    # or use your own visualization approach:
    # plotmodel(mdgs, data=mdgsV, ... ) # This is an example call
    pass

print("Done with the MISMIP script in Python.")
