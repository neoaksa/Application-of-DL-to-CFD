# Application-of-DL-to-CFD
* [Journal.md](https://github.com/neoaksa/Application-of-DL-to-CFD/blob/master/Journal.md): Some basic knowlege about OpenFOAM and DP for this project
* Bugs.md: to record issues
* [Progress.md](https://github.com/neoaksa/Application-of-DL-to-CFD/blob/master/Progress.md): project schedule and issues lists
* [ConvLSTM](https://www.jie-tao.com/convolutional-lstm/)
* [PredNet](https://coxlab.github.io/prednet/)
* [Final Report](https://github.com/neoaksa/Application-of-DL-to-CFD/blob/master/Application%20of%20Deep%20Learning%20to%20CFD.pdf)

## Structure
### Training
* Openfoam-prednet(t+1).py: Train for PredNet(t+1)
* Openfoam-prednet(t+n).py: Train for PredNet(t+n)

### Prediction
* Openfoam-prediction-prednet(t+1)w(t+1).py: output 1 timestep prediction through trained PredNet(t+1)
* Openfoam-prediction-prednet(t+n)w(t+10).py: output rolling 10 timestep prediction through trained PredNet(t+1) or PredNet(t+n)

### Validation
* Openfoam-validation-prednet(t+1)w(t+1).py: validation result for 1 timestep prediction through trained PredNet(t+1)
* Openfoam-validation-prednet(t+n)w(t+10).py: validation result for rolling 10 timestep prediction through trained PredNet(t+1) or PredNet(t+n)

### Model
* openfoam.h5: PredNet(t+1) 
* openfoam_stepforward.h5: PredNet(t+n)

## Result
* Application of Deep Learning to Computational Fluid Dynamics.docx
* Application of Deep Learning to Computational Fluid Dynamics.pptx
* Poster.pptx

## Thanks to:
Dr.Greg Wolffe
DEN: Distributed Execution Network
Roberto Sanchez

![poster](/Poster.jpg)
