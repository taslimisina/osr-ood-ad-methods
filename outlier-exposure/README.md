# Outlier Exposure

The codes are adapted from [the official repo for Outlier Exposure](https://github.com/hendrycks/outlier-exposure).

In this section, you can train a classifier, fine-tune with outlier exposure, train a model along with outlier exposure,
or evaluate a trained model for Out-of-Distribution detection. 

##how to run

For each inlier dataset, run the code in the corresponding folder. Below are examples for CIFAR-10. Other datasets are similar.

<br>
For training a classifier:

~~~
python baseline.py --dataset cifar10 --cifar10 /path/to/cifar10 --model wrn --save /folder/to/save/checkpoints
~~~


<br>
For fine-tuning with outlier exposure:

~~~
python oe_tune.py --dataset cifar10 --cifar10 /path/to/cifar10 --tinynet /path/to/tinynet --model wrn --save /folder/to/save/checkpoints 
~~~


<br>
For training from scratch with outlier exposure:

~~~
python oe_scratch.py --dataset cifar10 --cifar10 /path/to/cifar10 --tinynet /path/to/tinynet --model wrn --save /folder/to/save/checkpoints
~~~


<br>
For evaluation:

~~~
python test.py --num_to_avg 1 [-v] --method_name cifar10_wrn_baseline --load /path/to/model/checkpoint \
--ood_method ['MSP' | 'MLV' | 'Ensemble' | 'MC_dropout'] --cifar10 /path/to/cifar10 \
[--ens1 /path/to/ensemble/checkpoint1] [--ens2 /path/to/ensemble/checkpoint2] [--ens3 /path/to/ensemble/checkpoint3] \
--{dataset-name} /path/to/dataset
~~~
- params ens1,2,3 are for when "--ood_method" is set to "Ensemble"
- the {dataset-name} specifies outliers and you can enter multiple datasets. For cifar10, the choices are "cifar100", "texture", "svhn" "places365", "lsun". 