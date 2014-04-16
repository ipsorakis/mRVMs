-------------mRVMs README FILE ----------------
***** Psorakis Ioannis psorakis@gmail.com, University of Oxford, UK
*****Theodoros Damoulas damoulas@cs.cornell.edu, Cornell University, USA
============================================
1. What is this?

This documents acts as a manual to the matlab scripts that implement the ideas presented in 

Psorakis, I.   Damoulas, T.   Girolami, M. A.,"Multiclass Relevance Vector Machines: Sparsity and Accuracy", IEEE Transactions on Neural Networks, Vol 21, No 10, pp 1588 - 1598, 2010

The code base consists of:

-scripts for training:
train_mRVM1
train_mRVM2
train_mRVM1_MKL
train_mRVM2_MKL

-scripts for cross-validation (train and test):
cv_mRMV1
cv_mRMV2
cv_mRMV1_MKL
cv_mRVM2_MKL

-scripts for predictions:
predict_mRVM
predict_mRVM_MKL

where MKL stands for Multi-Kernel Learning and therefore problems with data from various sources/feature spaces.

scripts are ran under the MATLAB development environment (any other compatible environment such as Octave has *not* been tested)

you can do

>> help <script_name>

at any time to access information on how to use the scripts.

2. Datasets
In order for the above scripts to run, a dataset is needed which represents a specific instance of a supervised learning phenomenon. For our algorithms, the dataset should be in a ".mat" MATLAB workspace file, containing the following variables:


- the actual observations X with dimensions N X D, where N number of samples and D number of features.
- the target values t with dimensions C X N, where C number of classes. For example, if the 5th sample belongs to the 3rd class of a 5-class problem, then in the "t" matrix, the 5th column is as follows:
t = [
0
0
1
0
0]
============================================

3. Model Training

Model training consists of 4 scripts. 2 for mRVM1 and mRVM2 in the single kernel setting:

train_mRVM1.m
train_mRVM2.m 

and another 2 again for mRVM1 and mRVM2 for the multi-kernel setting:

train_mRVM1_MKL.m
train_mRVM2_MKL.m 

----------

You can use the scripts in two ways:
i) function argument passing
in MATLAB, "cd" in the directory where the scripts reside and type:

>> OUTPUT = train_mRVM1('-p',X,t,standardize_flag,convergence_used,kernel_type,kernel_param,plot_flag,dataset_name)

where the first argument '-p' denotes that the inputs are read as function arguments. Those are:
X: of size N x D (N samples, D features) is the training set.
t: of size C x N (C classes, N samples) is the training labels.
standardize_flag: [boolean] turns data standardization ON/OFF
convergence_used: values [1 or 2] is the training termination criterion (see
                  conv.1 and conv.2 of theoretical background
kernel_type: string can be either 'gaussian', or 'polynomial' or 'linear;
kernel_param: for linear kernel put any value
plot_flag: [1 or 0] plots the number of relevant vectors during training
dataset_name: auxiliary label
In case you use '-i' please note:
each dataset file must contain the necessary variables, which much be in the correct format. Those are:

1 the class labels, say "t". This variable must be a C X N dimensional array
  where C is the number of classes and N is the size of the data. Say we have a
  problem with 100 samples and 3 classes. The 5th sample would belong to the 2nd
  class if t(:,5)' = [0 1 0].
  For datasets which do have independent training and test sets, there should be two of these
  variables. E.g. tTrain and tTest

2 the data, say "X". This variable should be a N X D where N is the number of samples
  and D is the number of features. For datasets which do have independent training
  and test sets, there should be two of there variables. E.g. Xtrain Xtest. Also, for multi-kernel problems
  there should be one such variable for each feature space.

OUTPUT is an object that has a range of properties:
model_used: the name of the algorithm (e.g mRVM1);
dataset_name: the name of the dataset;
N_total: the total number of samples in the original dataset;
N_prototypical: the number of relevance vectors extracted from the algorithm;

X_prototypical: the relevance vectors (or prototypical samples) extracted from the algorithm;

X_prototypical_standardized: same as above, but standardized;
K_sparse: the sparse training kernel, that uses only the relevance vectors;
W_sparse: the sparse regressors matrix, that uses only the relevance vectors;

active_sample_original_indices: the original indices of the relevance vectors in the dataset;

sources: number of kernels;
b: kernel mixing coefficients;

kernel_type: kernel(s) used;
kernel_param: kernel parameter(s) used;


ii) command interface
in MATLAB, "cd" in the directory where the scripts reside and type:

>> OUTPUT = train_mRVM1('-i')

where the flag "-i" denotes command line interface. This asks the user to provide a series of information on the input variables needed to train the algorithm. The variables must be in the form described above.

at any point type help <script_name> to get this help information!
============================================

4. Prediction of new values

After the training phase, we assume that the user has the OUTPUT object with all the results. In the command line interface, we type:

>> [class_memberships accuracy] = predict_mRVM(OUTPUT,Xtest,test_labels)

where
Xtest: the test data of size Ntest x D where D the number of features SAME with the feature number of Xtrain.
test_labels: OPTIONAL variable or size C x Ntest where C the number of classes.

OUTPUTS:
class_membership_probabilities: of size Ntest x C, where the element
              (i,c) describes the degree of belief that sample-i belongs to class-c.
class_recognition_accuracy: IF test_labels are provided, tests the
                  accuracy of the algorithm based on the values of
                  class_membership_probabilities. Otherwise is NaN


============================================

5. Cross-validation

The "cv_*" scripts allow the user to perform K-fold cross-validation using different portions of the same data set for training and testing. The interface is exactly the same, both in command line and function argument interface, with the only difference that the user has to provide the K number of folds.

For example, say we want to perform 10 times cross validation using mRVM1 we write on the MATLAB command line:

>> cv_mRVM1('-p',10,X,t,1,'gaussian',.25,1,'iris')

where '-p' denotes that we pass the input as function arguments, "10" stands for 10-fold validation, X is our training data, t are the data labels, 'gaussian' is the type of kernel, .25 is the kernel parameter, 1 turns plotting ON, and "iris" is the name of the dataset.

Similar arguments can be passed by writing:

>> cv_mRVM1('-i')

and going through the command line dialogues.


============================================
IMPORTANT:

don't forget to download the script "YTruncate.m" which is distributed SEPARATELY from 
our website due to different licensing. It is vital for the execution of our code.
============================================

Thank you very much for your interest. We welcome all kinds of feedback and suggestions
-please contact the authors.

============================================

   Copyright NCR, 2009

   Use of this code is subject to the Terms and Conditions of the Research License Agreement,
   agreed to when this code was downloaded, and a copy of which is available at
   http://www.dcs.gla.ac.uk/inference/pMKL/Terms_and_Conditions.html.

====================================================
