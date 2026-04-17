import numpy as np
from SetIceShelfBC import SetIceShelfBC
from mismipbasalforcings import mismipbasalforcings

# Creating thickness
print('      creating thickness')
bx = -150 - 728.8*(md.mesh.x/300000)**2 + 343.91*(md.mesh.x/300000)**4 - 50.57*(md.mesh.x/300000)**6
by = (500./(1 + np.exp(-2./4000.*(md.mesh.y - 80000./2 - 24000))) +
      500./(1 + np.exp( 2./4000.*(md.mesh.y - 80000./2 + 24000))))
by0 = (500./(1 + np.exp(-2./4000.*(0     - 80000./2 - 24000))) +
       500./(1 + np.exp( 2./4000.*(0     - 80000./2 + 24000))))
md.geometry.bed     = np.maximum(bx + by, -720)
md.geometry.surface = np.maximum(bx + by0 + 100, 10)
md.geometry.base    = np.maximum(md.geometry.bed, -90)
md.geometry.thickness = md.geometry.surface - md.geometry.base

# Creating drag
print('      creating drag')
md.friction.coefficient = np.sqrt(3.160e6) * np.ones(md.mesh.numberofvertices)
md.friction.p           =            3 * np.ones(md.mesh.numberofelements)
md.friction.q           =            0 * np.ones(md.mesh.numberofelements)

# Creating flow law parameter
print('      creating flow law parameter')
md.materials.rheology_B   = 1/((6.338e-25)**(1/3)) * np.ones(md.mesh.numberofvertices)
md.materials.rheology_n   =            3 * np.ones(md.mesh.numberofelements)
md.materials.rheology_law = 'None'

# Boundary conditions for diagnostic model
print('      boundary conditions for diagnostic model')
md = SetIceShelfBC(md, './Front.exp')
md.mask.ice_levelset[:]   = -1
md.mask.ocean_levelset[:] = -1
pos = np.where((md.mesh.x < 640000.1) & (md.mesh.x > 639999.9))[0]
md.mask.ice_levelset[pos] = 0
md.stressbalance.spcvx[:]  = np.nan
md.stressbalance.spcvy[:]  = np.nan
pos = np.where(((md.mesh.y <  80000.1) & (md.mesh.y >  79999.9)) |
               ((md.mesh.y <      0.1) & (md.mesh.y >     -0.1)))[0]
md.stressbalance.spcvy[pos] = 0
pos2 = np.where((md.mesh.x <      0.1) & (md.mesh.x >     -0.1))[0]
md.stressbalance.spcvx[pos2] = 0
md.stressbalance.spcvy[pos2] = 0

# Forcing conditions
print('      forcing conditions')
# --- basal melt and buttressing conditions -----------------------
md.basalforcings = mismipbasalforcings()      #  ← remove md


md.basalforcings.meltrate_factor        =   0
md.basalforcings.threshold_thickness    =  75
md.basalforcings.upperdepth_melt        = -100
md.smb.mass_balance                    = 0.3 * np.ones(md.mesh.numberofvertices)
md.basalforcings.geothermalflux        = 0.5 * np.ones(md.mesh.numberofvertices)
md.basalforcings.groundedice_melting_rate = 0. * np.ones(md.mesh.numberofvertices)

# Other model parameters
md.thermal.spctemperature        = np.nan * np.ones(md.mesh.numberofvertices)
md.groundingline.migration       = 'SubelementMigration'
md.materials.rho_ice             = 918
md.materials.rho_water           = 1028
md.constants.g                   = 9.8
md.constants.yts                 = 31556926
md.transient.isthermal           = 0
md.transient.isgroundingline     = 1
md.stressbalance.isnewton        = 0

# Initialization
md.initialization.vx          = np.ones(md.mesh.numberofvertices)
md.initialization.vy          = np.ones(md.mesh.numberofvertices)
md.initialization.vz          = np.ones(md.mesh.numberofvertices)
md.initialization.vel         = np.sqrt(2) * np.ones(md.mesh.numberofvertices)
md.initialization.pressure    = md.constants.g * md.materials.rho_ice * md.geometry.thickness
md.initialization.temperature = 273 * np.ones(md.mesh.numberofvertices)
