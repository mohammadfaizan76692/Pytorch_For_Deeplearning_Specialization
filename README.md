# Pytorch_For_Deeplearning_Specialization
It is the course teach by Laurence Moroney on Deep learning AI.

## Pytorch FundatMentals
Module1 : Getting Started with Torch  (torch basic, why torch)  

Module2 : The pytroch WorkFLow (Deeplearning Model training flow zero_grad, forwardpass, loss, backward pass, weights update)  

Module3 : Data Management in Pytorch (Custom Dataset , Custom Dataloader , Error Logs , sikking bad samples , stats of sample picking) 

Module4 : Core Neural Network Components

### Summary by Me.
1) creating simple Neural Nets.
2) Actual flow of Training.
3) Custom DataLoader (importance of Shuffling) , Dataset, transforms from torchvision 
4) What is ToTensor(), Normalize(mean, std) for each channel helps model to learn better Siganl/information from  data.
5) Batch Normalization: Make training faster and stable.
6) CNN , kernel , padding, stride, Pooling
7) Dynamic Flow of Pytorch, nn.Sequential()  
8) loss function, gradient, optimizer, backward prop
9) Regularization by : weight_decay, dropout layer, batchnormalization
10) Error logs, skipping bad example while picking and stats pf samplings

## Pytorch Techinques and EcoSystem Tools:
Module1:  
currently learn : Hyper parameter Tuning : Batch size, Learning Rate  

In Lab1
Metrics: torchmetrics (pip install torchmetrics) -> create , update, calculate Micro/Macro[Preciseion, Recall, F1 Score]    
  
In Lab2: learned Learning rate scheduler ,  
StepLR: Gamma: factor of rate decay, step size: Number of Epochs (after that ,lr = lr*Gamma), optimizer: defined optimizer  
ReduceLROnPlateau: optimizer, mode =max or min, factor: by which it reduced , patience = how much step we have to weight before updating it.  
CosineAnnealingLR: decreasing like cosine wave, optimizer,  Tmax= total epochs, eta_min = 0.0002 -> mininum value can lr have  
  
In Lab3:  
1)How to Define Flexible architecture where (Number of layers, Number of filters, kernel size, dropout value, linear layer)  
  
2) Optuna for selecting best hyperparameter using TPE ( Tree-structured Parzen Estimator)  
Creating objective function which take trial (trial,.....) and  parameters of model training , trial.suggest () suggesting the hyperarameter using those hyperparameter whole model train  
and evaluate on validation set and return metric which you want to optimize.  

3) Creating optuna study , and optimize study with number to trials and objective function.  
   for More refer ugraded lab3 in Module1  

In Lab4:  







