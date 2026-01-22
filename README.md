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

In Lab2:
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
In Lab3:
torch visions, utils functions
1) utiltiy for Annotation

  Drawing Bonding boxes for task like object detection** 
  
      ```python
      from# Pytorch_For_Deeplearning_Specialization
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

In Lab2:
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
In Lab3:
torch visions, utils functions
1) utiltiy for Annotation

Drawing Bonding boxes for task like object detection** 
  
      ```python
      from torchvision import utils as vutils
      result = vutils.draw_bounding_boxes(image=image, 
                                          boxes=boxes, 
                                          labels=labels,           # This is optional
                                          colors=["red", "blue"],  # This is optional. By default, random colors are generated for boxes.
                                          width=3                  # This is optional. The default is width=1
                                    )
      ```

Drawing Segmentation MasK  

      ```python
      ## object mask  is of shape (num_of_masks, h,w) , values are true and false, and true value will be masked with given color
      result  = vutils.draw_segmentation_masks(image=image,
                                          masks=object_mask,
                                          alpha=0.5,          # This is optional. The default is alpha=0.8
                                          colors=["blue"]     # This is optional. By default, random colors are generated for each mask.
                                          )
      ```
                                        
                                        
2) Already Available trained Architectures for Different Vision task can be Used 
	1) If task is similar can directly Use them  : (Inference (Out-of-the-Box Prediction)
	2) If task are not similar, Will Do Fine Tuning, using Model learning initial learning like edge and feature detection and  finetune for our purpose (Transfer learning)
	
	Some Example Models
	Image Classification: Answers the basic question: 'What is the main subject of this image?'
	Models: ResNet, VGG, AlexNet, SqueezeNet, MobileNetV3, DenseNet
	Image Segmentation: Goes deeper to ask: 'What is the exact pixel-by-pixel shape of each object?'
	Models: FCN, DeepLabV3
	Object Detection: Finds all recognizable objects in an image, draws a box around each one, and classifies them.
	Models: Faster R-CNN, RetinaNet, SSD
	Video Classification: Understands action and movement by classifying entire video clips.
	Models: R(2+1)D 18, MC3 18, Video MViT
	
	3) Reading classes and names on Which that model have trained using .meta attributes of weights of Model

In Lab4: 
Transfer Learning Strategies
Using a pre-trained model for inference is powerful, but its true potential is unlocked when you adapt it for a new, custom task. This process, called transfer learning.
Three Ways of Transfer learning
1) Feature Extraction: Freezing the model's backbone and training only the new classifier head.
      Extracting classifier head can be Modular or attribute of Model, and change it according  to you class.
      In optimizer will pass only those weights where required_grad = True

2) Fine-Tuning: Unfreezing and training the top layers of the backbone for better adaptation.
            Unfreeqing = making required_grad  = True

3) Full Retraining: Training the entire model end-to-end for maximum performance.
      retrain the whole parameters after changing classifier layer (compuational expensive)


Graded Assignment: Imporving Fake Image Finder Using Transfer Learning Strategies

Improving FakeImage Finder

1) Using ImageFolder for Dataset creation.
2) Create Custom transforms.
3) Apply Transfer learning ("feature Extraction") Using MobileNetV3-Large architecture.

Module3: 
NLP with Pytorch and Hugging Face
In Lab1:
Tokenizers
1) Creating simple Vocabulary.
2) Pretrained Tokenizer, BERT (Bi-Directional Encoder Representation from Transformers)  [BertTokenizerFast].
3) Padding:  <PAD> adding to tokens to cosistent length, Truncation: trucating the sentence or document to make consistent length.
4) tokenizer.convert_ids_to_tokens : converting index to tokens.
5) tokenizer.get_vocab() # to gt the whole vocabulary.
6) Attention Mask: [1, 1, 1, 0, 0] => 0 means it is padding no need of attention while processing that sentence in Model.
7) AutoTokenizer: No Need To define each and every Model Tokennizer , just pass tokenizer from hugging face or downloaded it automatically Handled That.
8) OOV: Out of Vocabulary Words ( Not in Vocubulary/built in dictionary) : Handle by splitting in small sub words , Musbi become ['mu', '##bs', '##i'] , ## mean they are part of one Whole Word.

	
In Lab2:
Tensorization : one hot encoding, bag of word(frequency count), tf -IDF,  Embeddings [Static Embeddings model like (glove) and Dynamic Embeddings (BERT)]
1) Visualizing Glove Pretrained Embedding Relationship like : king - man + woman ~ queen
2) Creating Embeddings for self Dataset Using nn.Embeddings 
	creating similar word pairs  (words that sould be in one Category)
	in pair taking one as input word, other as target word 
	(input word, target word) - (input word - index ) -> embedding vector of nn.Embedding(Vocab_size, vector_size) ->  target word prediction vocubulary size final layer.
	
	training using CrossEntropy like classification.
	
3) Dyanmic vs Static 
	bat animal 
	baseball bat  
	In static both are same embedding vectors : Visualized from Glove
	but dynamic embeddings are different : visualized from BERT


In Lab3:
Text Classification
1) Text preprocessing, label processing, train_test_split to avoid data leakage.
2) Vocabulary creation Using Python Class, with '<pad>','<unk>' token, word2idx() And idx2word(), encode(), build_vocab(), with min freq tokens will go into vocabulary.
3) Custom Text-Dataset
4) DataLoaders By collate function 
	collate_fn is also the standard and most efficient method. While you could pad all sentences from the start, that approach is inefficient, leading to significant memory and computational waste. 
	By using a collate_fn to perform "dynamic padding," each batch is only padded to the length of the longest sentence within that batch, which saves resources and speeds up training.
	1) collate_batch_embeddingbag : [flatten_text, offest = starting index of each sentence in flatten_text, labels]
		in these batches using - nn.EmbeddingBag
	2) collate_batch_manual : [padded_text with max length in batch, labels]
		in these batches manual pooling and nn.Embedding
		 
		
5) Comparing Manual : max, mean, sum  and nn.EmbeddingBag  | | | Best Lab Till Now | | | 


In Lab4:
Pretrained for same dataset in lab3
1) How Saving transformer model tokenizer and model
2) loading model and tokenizer
3) Creating Custom Dataset with bert tokenizer
4) Data collator handles dynamic padding for each batch
      ```python
       	data_collator = transformers.DataCollatorWithPadding(tokenizer=bert_tokenizer)
	# Create the DataLoader for the training set with `data_collator`
	train_loader = DataLoader(train_dataset, 
		                  batch_size=batch_size, 
		                  shuffle=True, 
		                  collate_fn=data_collator
		                 )
	```
5) class weight from sklearn : from sklearn.utils.class_weight import compute_class_weight
	```python
	# Extract all labels from the training set to calculate class weights for handling imbalance.
	train_labels_list = [train_dataset.dataset.labels[i] for i in train_dataset.indices]
	    
	    
	# Use scikit-learn's utility to automatically calculate class weights.
	class_weights = compute_class_weight(
	    # The strategy for calculating weights. 'balanced' is automatic.
	    class_weight='balanced',
	    # The array of unique class labels (e.g., [0, 1]).
	    classes=np.unique(train_labels_list),
	    # The list of all training labels, used to count class frequencies.
	    y=train_labels_list
	)

	# Convert the NumPy array of weights into a PyTorch tensor of type float
	class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
	# Initialize the CrossEntropyLoss function with the calculated `class_weights`.
	loss_function = nn.CrossEntropyLoss(weight=class_weights)
	
	```
6) Instead of full retraining have done finetuning only using last 2 transformer_layers

7) comparing full retraining(full finetuning) and fine tuning(partial fine tunining) (Transfer learning Method)
 torchvision import utils as vutils
      result = vutils.draw_bounding_boxes(image=image, 
                                          boxes=boxes, 
                                          labels=labels,           # This is optional
                                          colors=["red", "blue"],  # This is optional. By default, random colors are generated for boxes.
                                          width=3                  # This is optional. The default is width=1
                                    )
      ```

   Drawing Segmentation MasK  

      ```python
      ## object mask  is of shape (num_of_masks, h,w) , values are true and false, and true value will be masked with given color
      result  = vutils.draw_segmentation_masks(image=image,
                                          masks=object_mask,
                                          alpha=0.5,          # This is optional. The default is alpha=0.8
                                          colors=["blue"]     # This is optional. By default, random colors are generated for each mask.
                                          )
      ```
                                        
                                        
2) Already Available trained Architectures for Different Vision task can be Used 
	1) If task is similar can directly Use them  : (Inference (Out-of-the-Box Prediction)
	2) If task are not similar, Will Do Fine Tuning, using Model learning initial learning like edge and feature detection and  finetune for our purpose (Transfer learning)
	
	Some Example Models
	Image Classification: Answers the basic question: 'What is the main subject of this image?'
	Models: ResNet, VGG, AlexNet, SqueezeNet, MobileNetV3, DenseNet
	Image Segmentation: Goes deeper to ask: 'What is the exact pixel-by-pixel shape of each object?'
	Models: FCN, DeepLabV3
	Object Detection: Finds all recognizable objects in an image, draws a box around each one, and classifies them.
	Models: Faster R-CNN, RetinaNet, SSD
	Video Classification: Understands action and movement by classifying entire video clips.
	Models: R(2+1)D 18, MC3 18, Video MViT
	
	3) Reading classes and names on Which that model have trained using .meta attributes of weights of Model

In Lab4: 
Transfer Learning Strategies
Using a pre-trained model for inference is powerful, but its true potential is unlocked when you adapt it for a new, custom task. This process, called transfer learning.
Three Ways of Transfer learning
1) Feature Extraction: Freezing the model's backbone and training only the new classifier head.
      Extracting classifier head can be Modular or attribute of Model, and change it according  to you class.
      In optimizer will pass only those weights where required_grad = True

2) Fine-Tuning: Unfreezing and training the top layers of the backbone for better adaptation.
            Unfreeqing = making required_grad  = True

3) Full Retraining: Training the entire model end-to-end for maximum performance.
      retrain the whole parameters after changing classifier layer (compuational expensive)


Graded Assignment: Imporving Fake Image Finder Using Transfer Learning Strategies

Improving FakeImage Finder

1) Using ImageFolder for Dataset creation.
2) Create Custom transforms.
3) Apply Transfer learning ("feature Extraction") Using MobileNetV3-Large architecture.

Module3: 
NLP with Pytorch and Hugging Face
In Lab1:
Tokenizers
1) Creating simple Vocabulary.
2) Pretrained Tokenizer, BERT (Bi-Directional Encoder Representation from Transformers)  [BertTokenizerFast].
3) Padding:  <PAD> adding to tokens to cosistent length, Truncation: trucating the sentence or document to make consistent length.
4) tokenizer.convert_ids_to_tokens : converting index to tokens.
5) tokenizer.get_vocab() # to gt the whole vocabulary.
6) Attention Mask: [1, 1, 1, 0, 0] => 0 means it is padding no need of attention while processing that sentence in Model.
7) AutoTokenizer: No Need To define each and every Model Tokennizer , just pass tokenizer from hugging face or downloaded it automatically Handled That.
8) OOV: Out of Vocabulary Words ( Not in Vocubulary/built in dictionary) : Handle by splitting in small sub words , Musbi become ['mu', '##bs', '##i'] , ## mean they are part of one Whole Word.

	
In Lab2:
Tensorization : one hot encoding, bag of word(frequency count), tf -IDF,  Embeddings [Static Embeddings model like (glove) and Dynamic Embeddings (BERT)]
1) Visualizing Glove Pretrained Embedding Relationship like : king - man + woman ~ queen
2) Creating Embeddings for self Dataset Using nn.Embeddings 
	creating similar word pairs  (words that sould be in one Category)
	in pair taking one as input word, other as target word 
	(input word, target word) - (input word - index ) -> embedding vector of nn.Embedding(Vocab_size, vector_size) ->  target word prediction vocubulary size final layer.
	
	training using CrossEntropy like classification.
	
3) Dyanmic vs Static 
	bat animal 
	baseball bat  
	In static both are same embedding vectors : Visualized from Glove
	but dynamic embeddings are different : visualized from BERT


In Lab3:
Text Classification
1) Text preprocessing, label processing, train_test_split to avoid data leakage.
2) Vocabulary creation Using Python Class, with '<pad>','<unk>' token, word2idx() And idx2word(), encode(), build_vocab(), with min freq tokens will go into vocabulary.
3) Custom Text-Dataset
4) DataLoaders By collate function 
	collate_fn is also the standard and most efficient method. While you could pad all sentences from the start, that approach is inefficient, leading to significant memory and computational waste. 
	By using a collate_fn to perform "dynamic padding," each batch is only padded to the length of the longest sentence within that batch, which saves resources and speeds up training.
	1) collate_batch_embeddingbag : [flatten_text, offest = starting index of each sentence in flatten_text, labels]
		in these batches using - nn.EmbeddingBag
	2) collate_batch_manual : [padded_text with max length in batch, labels]
		in these batches manual pooling and nn.Embedding
		 
		
5) Comparing Manual : max, mean, sum  and nn.EmbeddingBag  | | | Best Lab Till Now | | | 


In Lab4:
Pretrained for same dataset in lab3
1) How Saving transformer model tokenizer and model
2) loading model and tokenizer
3) Creating Custom Dataset with bert tokenizer
4) Data collator handles dynamic padding for each batch
      ```python
       	data_collator = transformers.DataCollatorWithPadding(tokenizer=bert_tokenizer)
	# Create the DataLoader for the training set with `data_collator`
	train_loader = DataLoader(train_dataset, 
		                  batch_size=batch_size, 
		                  shuffle=True, 
		                  collate_fn=data_collator
		                 )
	```
5) class weight from sklearn : from sklearn.utils.class_weight import compute_class_weight
	```python
	# Extract all labels from the training set to calculate class weights for handling imbalance.
	train_labels_list = [train_dataset.dataset.labels[i] for i in train_dataset.indices]
	    
	    
	# Use scikit-learn's utility to automatically calculate class weights.
	class_weights = compute_class_weight(
	    # The strategy for calculating weights. 'balanced' is automatic.
	    class_weight='balanced',
	    # The array of unique class labels (e.g., [0, 1]).
	    classes=np.unique(train_labels_list),
	    # The list of all training labels, used to count class frequencies.
	    y=train_labels_list
	)

	# Convert the NumPy array of weights into a PyTorch tensor of type float
	class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
	# Initialize the CrossEntropyLoss function with the calculated `class_weights`.
	loss_function = nn.CrossEntropyLoss(weight=class_weights)
	
	```
6) Instead of full retraining have done finetuning only using last 2 transformer_layers

7) comparing full retraining(full finetuning) and fine tuning(partial fine tunining) (Transfer learning Method)
