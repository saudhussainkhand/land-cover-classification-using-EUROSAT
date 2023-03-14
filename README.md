# Land-Cover-Classification-using-EUROSAT

![EUROSAT](https://user-images.githubusercontent.com/60270854/224863614-e7d72936-ad78-4103-83dc-7a3a805d5261.jpg)

# Dataset Description
* EuroSAT → Contains RGB images collected from the Sentinel Data having 20 classes.
  * AnnualCrop
  * Forest
  * HerbaceousVegatation
  * Highway
  * Industrial
  * Pasture
  * PermanentCrop
  * Residential
  * River
  * SeaLake

* Dataset is available at https://drive.google.com/file/d/1G6V8wncoNYP0DmNIU0k92uNdWamlOJcN/view


# Approach Followed
* Loaded the data 
* Exploratory Data Analysis
* Preprocessing 
  * Dividing data into training and testing folders (80,20)
  * Saving class labels
* Defining the model
  * Fine-tuning vgg-19 model
* Training the model
  * Data Loaders Used
  * Num_epoch=100 with early stopping so that model doesnot overfit
  * lr=0.01 (learning rate)
  * loss=categorical_crossentropy
* Results and Predictions (Inference Notebook contains the predictions and other model stats)
 ![download](https://user-images.githubusercontent.com/60270854/224865619-ccf59fba-f8dd-4d80-9797-9539d07c9aee.png)
 ![results](https://user-images.githubusercontent.com/60270854/224866035-9d28ec4b-8fb4-4707-b5d1-9701a71d680b.png)

  
* Save the model
 * Model: https://drive.google.com/file/d/1-83an_jtfHn-OGvjQMv7YyHG3JKnHOrI/view?usp=sharing
 
 
 
 
# Constraints and Potential Improvements to current Solution
* The current solution yeild an accuracy score of 96.9% but there are certain ways in which we can improve this solution and acheive much more better and efficient results.
* Data Augmentation: Data augmentation is useful to improve the performance and outcomes of machine learning models by forming new and different examples to train datasets. If the dataset in a machine learning model is rich and sufficient, the model performs better and more accurately. The EUROSAT dataset is slightly imbalanced using data augmentations can help us improve the accuracy and achieve better results.
* Adding Reduced Learning Rate: Reduce learning rate when a metric has stopped improving. This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.
* Adding Gradient clipping: It can prevent vanishing and exploding gradient issues that mess up the parameters during training. In order to match the norm, a predefined gradient threshold is defined. Gradient norms that surpass the threshold are reduced to match the norm. The norm is calculated over all the gradients collectively, and the maximum norm is 0.1.
* Transfer Learning Approach: In my solution I have fined tuned the model on vgg19. We can try and work with Resnet pretrained, Inceptionv6 and other pretrained networks and evaluate the performance of the model to see if it improves more than our current scores.
* Increasing number of dense layers: In my solution I have worked with 512 dense layers. We can increase the number of dense layers with vgg19 to 1024 and 2048 and in many cases has provided with better results.
* Time to Train: This was one the major issues faced while training the model. The model took 3hours to first epoch with a batch size of 64. The time to train is on of the constraints of this solutions.
* Memory Limitations: The developed solution creates the training and testing directories seperately which can require additional memory usage. The best weights being stored in callbacks and model saving also consumes memory.
* GPU Timeout: The model requires a lot of time to train, leading to GPU timeout mostly, if we wish to train multiple models and compare their perforamnces on this large dataset, you need to have GPU subscription else the GPU gets timedout (happened alot in my case).


# Summarizing the constraints and potential improvements
* Constraints
 * Training Time
 * Data is imbalance
 * Limited to one model
 * Memory Limitations

* Potential Improvements
  * Data Augmentation
  * Adding Gradient Clipping 
  * Adding Reduced LR 
  * More complex model
  * Try different models and approaches such a transfer learning 
  
  

