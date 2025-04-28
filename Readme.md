# Course - Machine Learning and its Applications in Aerospace Power
by Chensheng Luo in the spring semester, 2025

**Machine Learning and its Applications in Aerospace Power** is a graduate course of Beihang University, taught by Assoc. Prof. FAN yu and Prof. QIU Lu. This repository contains my training code, homework, self-made application *etc.* for this course.

## Respo contents
- [testpython](./testpython/): basic study of python and package version test.
- [Classification-KNN-SVM](./Classification-KNN-SVM/): Classification problem using K-Nearest Neighbour and Support Vector Machine
- [ANN](./ANN/): Artificial Neural Network
    - self-made ones: [ANN class](./ANN/ANN_selfmade.py), [example](./ANN/ANN_selfmade_test.py) 
    - using `pytorch`: 
    - using `tensorflow`: [classification example](./ANN/ANN_tensorflow_test.py)
- [PINN](./PINN/): Physical Informed Neural Network

    All done using `pytorch`
    - Heat Equation function: [FDM solver(classical)](./PINN/HeatEquationFDM.py), [PINN solver](./PINN/HeatEquationPINN.py)
    - [Cylinder Flow](./PINN/CylinderFlowPINN.py) 