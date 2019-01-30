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

~~1. How to use(or choose peak) frames in CNN~~
~~2. How to create samples automatically~~
~~3. Know How solver works in the different situations.~~

1. understand the data meaning inside the frame files
2. change the velocity and angle  of flow also the size of object
3. simplify the animation, try "matplotlib.pyplot.quiver "
4. find if any stable status in the simulation
5. try to figure out the error of triangle issue

### week3
Done:
1. Automatic generate samples according to the object shape and angle of flow on seawolf.
  * Incoming flow angle is set from 0 degree to 90 degree with a step of 15 degree.
  * Shape is configured under the script `run` in each subfolder. [Example](/Samples/square_(-3%2C-3)_(3%2C-3)_(-3%2C3)_(3%2C3)_circle/run)
  * Samples generation can be found [here](/Samples)
  * For automatically generate all sampels, `runlist` has be set to call all `run` scripts under each subfolder.
2. Delete the temp files, only keep final frames
3. All samples can be found on seawolf, structure as follows:
`shape folder / angle folder / paraview folder`

Issue:
1. Can not call pravaview GUI through docker
2. Most of samples cannot reach a stable status.See [here](/img/square_3_3_1_45.png)
3. Number of samples are small(36), but number of frames are large(36\*150)

Next:
1. Label the samples
2. Export Vector files automaticaly
3. LTSM
