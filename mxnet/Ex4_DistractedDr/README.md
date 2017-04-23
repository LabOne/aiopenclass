

This path includes a solution for the Kaggle problem, State Farm Distracted Driver Detection https://www.kaggle.com/c/state-farm-distracted-driver-detection. This example illustrates how a CNN can be used to solve a image classification for a practical problem with high accuracy. 


Ex4_distracted_drivers_cpu.ipynb: the major notebook file using CPU describing every step for this problem. 
Ex4_distracted_drivers_gpu.ipynb: the major notebook file using GPU describing every step for this problem.

The python files include the following codes: 

1)dataSep.py: read the origical image files and separate them into the training and test datasets and save both respectively. You can use the command to setup the parameters including path: python  dataSep.py --INPUT ......
Use python dataSep --help to know more ; 

2)cnndd.py: library of CNN definition for this problem;

3) train.py: training code ; 

4) model_eval.py: the evaluation of the model using various metrics such as pd, accuracy, precision and mloglikelihood;

5) test_random.py: randomly select several validation photoes to test the CNN; 

6) fullymodel-0050.params and fullymodel-symbol.json: files for a pretrained model using AWS for this solution. 
