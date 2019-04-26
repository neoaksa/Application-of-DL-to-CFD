# Abstraction
The modeling of complex physical and biological phenomena has long been the domain of computational fluid dynamics.  Beginning with the Navier-Stokes equations describing fluid motion, numerous solvers have been derived and applied to various time-based modeling problems.  Given the recent success of deep learning models in a variety of application areas, this project attempted to determine if a deep neural network could be used to predict fluid motion.
The neural net architecture employed in this research was the ConvLSTM. It used a convolutional layer to discover spatial features, and a Long Short-Term Memory layer to learn time-based patterns.  It contained almost 7 million parameters, which took about 100 minutes to train on Ghost (Titan V GPU w/5120 cores). Two models were trained: Prednet(t+1) (predict next frame), and Prednet(t+10) (predict next 10 frames). Using these models, three prediction strategies were developed.
Results indicate that two of the strategies generate predictions visually indistinguishable from ground truth (defined as the results obtained using the Pisofoam incompressible flow solver).  Standard video prediction metrics (Peak Signal to Noise Ratio) showed that the Prednet(t+10) model was able to extrapolate rolling predictions for 10 seconds with reasonable accuracy in a controlled environment. Further experiments and training with a variety of samples (velocity, mesh, features) would be needed to determine if the models have the ability to generalize over a wide range of conditions.	


# Application-of-DL-to-CFD
* [Journal.md](https://github.com/neoaksa/Application-of-DL-to-CFD/blob/master/Journal.md): Some basic knowlege about OpenFOAM and DP for this project
* Bugs.md: to record issues
* [Progress.md](https://github.com/neoaksa/Application-of-DL-to-CFD/blob/master/Progress.md): project schedule and issues lists
* [ConvLSTM](https://www.jie-tao.com/convolutional-lstm/)
* [PredNet](https://coxlab.github.io/prednet/)
* [Final Report](https://github.com/neoaksa/Application-of-DL-to-CFD/blob/master/Application%20of%20Deep%20Learning%20to%20Computational%20Fluid%20Dynamics.pptx)

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
