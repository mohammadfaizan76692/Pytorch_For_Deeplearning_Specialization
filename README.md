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
Deployment metrics : Inference Time, Memory foot print( model size) , Model performance Metric
using defferent model find all the above values for each model  
Inference time : warm jsut run model for some few examples, then run the model n times with the data , then sum time take average_time = total_time/n
Memory : [params*params_size from model.parameters()  +  buffer*buffer_size from model.buffers()]//(1024**2) # byte to mb
Model performance Metric : Metric for which you are optimizing your model

There are Two types you can select best model for your need  
1) Contraint based : Removing all the model which does lie under these contraint and choose model which have high accuracy.  
2) Weighted score based : first defined weightage for all metric you want than normalizing all the metric of  Deployment  
          [0,1] -> val-min_val/(max_val-min_val) , for metric you want to decrease = 1 - ((val -min_val)/(max_val- min_val))    


Graded Assignment : Developed AI/Fake Images Detector.
1) Using FlexibleCNN architecture:  
      HyperParameters : number_of_layers, kernel_size for each layer, no_of_filters for each layer, linear_size  for classifier_layer, batch_size, Resolution(for cropping).   

2) Defining Search space for HyperParameters using Trial.suggest().   
  
3) Objective function : for optimizing the model with metrics.  

4) Efficiency Metrics: calculating Number of Train of Trainable Parameters for Each Model and corresponding Accuracy.  

5) Visualizing How Number of trainable Parameters for each model effecting Accuracy.

6) In option Also adding Precision to Reduce FP, and Recall to Reduce False Negative  and then visualizing  which model has best Precision and Recall


Module2:
Learning Vision Specific Support In torch , specifically torchvision


In Lab1:
Torch Vision Transforms
1) Resize : Resizing : contracting and stracehing image ,... Crop : cutting image -> For making consistent image Data  
2) for Augmentation RandomResizeCrop, Random flips and colorJitter and many more  
3) ToTensor() -> PIL to tensor , heightXwidth -> (c, height, width ), with normalized pixel values 0 to 1  
4) Normalize(mean[], std[]), in each column to shifting mean to zero and std to 1   
5) decode_image -> direct read any image as to tensor from torchvision.io    
6) make_grid -> taking batch of images, and show them in grid way from torchvision.utils    
7) save_image -> taking tensor image and save at specified path    
8) calculating mean and std for each channel in dataset  
9) combining all transforms in pipeline using transforms.Compose()  
10) creating custom transform function , example salt pepper ,  randomly converting pixel zero to one  

In Lab2
torvision Dataset learning, prebuilt dataset loading and inspection
1) directly download vision datasets.
2) dataset.transform = defined_tansformed.
3) Some dataset like emnist have unique requirements, have to handled differently. (when downloading need to specify split).
4) for custom data, ImageFolder can be used but each subfolder is class and corresponding images. And will be used as Dataset
	fruit_dataset = datasets.ImageFolder(root=root_dir,
		                             transform=image_transformation
		                            )
5) Creating Fake Dataset for debugging Architecture
	Initialize the FakeData dataset 
      ```python
      from torchvision import datasets

      # Initialize the FakeData dataset
      fake_dataset = datasets.FakeData(
      size=1000,                     # Total number of fake images
      image_size=(3, 32, 32),        # (Channels, Height, Width)
      num_classes=10,                # Number of possible classes
      transform=fake_data_transform  # Apply the transformation
      )
      ```




