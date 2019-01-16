## week1
1. install OpenFoam| |Y| |@20181220| 
2. create Simple Mesh|BlockMeshDict , command “bl
3. boundary control|blockMeshDict|Y| |@20181228| 
4. structure of folders in OpenFoam| |Y| |@201901
5. fluid propeties|transportProperties|Y| |@20190
6. initial status|0 folder or setFiledsDict|Y| |@
7. slover selection| |N|don’t know how to select 
8. time control|controlDict|Y| |@20190102| 
9. region setting|setFiledsDic & command “setFile
10. run solution|command with solution name|Y| |@
11. visual and export|paraview foam.foam|Y| |@201
12. create polyhrel or complex mesh| |N|In proces
13. parallel computation in OpenFoam| |N|Low prio
14. set OpenFoam goal and simple demo| |N| | | 
15. Deep learning solution discovery| |N| | | 

### week2
Done:

1. Parallel computation in Openfoam
2. Simulate single shapes(square, cycle and trapezoidal) in the water flow.
3. Output vector plots.
4. create mesh with gmesh tools,  time costing.

Issue:

1. When I tried a triangle or two objects in the mesh, the error happen since computation couldn't reach convergence.
2. The files is too large, 1GB for 36 frames in a single simulate. This might be a problem when we create enough samples for CNN.

Next:

1. How to use(or choose peak) frames in CNN
2. How to create samples automatically
3. Know How solver works in the different situations.
