# Workspace analysis for multi-target electromagnetic control
---
We contain mainly kinds of examples:
1. Calculating multi-target magnetic/wrench/hybrid-feasible workspace.
2. Calculating the task-feasible workspace, with comparison between discretization based method and interval-analysis based method. 
3. A case study for calculating the task-feasible workspace for a movable magnetic where a fixed magnet exists in his workspace.
We also provide many function libraries for electromagnetic control as well as the affine/interval arithmetic functions.
---
## 1. Calculating different kinds of actuation-feasible workspace

In `AFW_2D.py`, we provide the code example to calculate the actuation-feasible workspace of two magnets. We can set the poses and some physical properties of the magnets as we need. For example,
`
'X': 0.02, 'Y': 0.02, 'Z': -0.03, 'm': 0.2, 'alpha': np.pi/2, 'beta': np.pi/2
`
It defines the position and orientation of the magnets, as well as the magnetic moment `'m'`.
We can also define the actuation elements that we want to use for calculating workspace. All the available elements are
`
'Bx': True, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 
         'fx': None, 'fy': None, 'fz': None, 'tx': None, 'ty': None, 'tz': None
`
If you want to see one specific magnetic actuation element, just set it as **True**. Otherwise, set it as **None**.
The result in our code example is 

<img src="images/AFW.png" alt="AFW" width="400">

We set the **Bx** at two actuation elements of two magnets that we want to get the workspace.

---
## 2. Compare two method to get the task-feasible workspace
In `Comparison.py`

<img src="images/Comparison.png" alt="Comparison" width="600">