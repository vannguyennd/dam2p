# DAM2P

The source code samples for reproducing the experimental results mentioned in our paper "Cross Project Software Vulnerability Detection via Domain Adaptation and Max-Margin Principle". Refer to http://arxiv.org/abs/2209.10406 for details.

## Datasets
We used the real-world datasets experimented in Nguyen et al. (2019, 2020). These contain the source code of vulnerable functions (vul-funcs) and non-vulnerable functions (non-vul-funcs) obtained from six real-world software project datasets, namely FFmpeg (#vul-funcs: 187 and #non-vul-funcs: 5427), LibTIFF (#vul-funcs: 81 and #non-vul-funcs: 695), LibPNG (#vul-funcs: 43 and #non-vul-funcs: 551), VLC (#vul-funcs: 25 and #non-vul-funcs: 5548), and Pidgin (#vul-funcs: 42 and #non-vul-funcs: 8268). These datasets cover multimedia and image application categories.

In the experiments, to demonstrate the capability of our proposed method in the transfer learning for cross-domain software vulnerability detection (SVD) (i.e., transferring the learning of software vulnerabilities (SVs) from labelled projects to unlabelled projects belonging to different application domains), the datasets (FFmpeg, VLC, and Pidgin) from the multimedia application domains were used as the source domains, whilst the datasets (LibPNG and LibTIFF) from the image application domains were used as the target domains. It is worth noting that in the training process we hide the labels of datasets from the target domains. We only use these labels in the testing phase to evaluate the models’ performance. Moreover, we used 80% of the target domain
without labels in the training process, while the rest 20% was used for evaluating the domain adaptation performance. We note that these partitions were split randomly as used in the baselines.

## Requirements 

We implemented our DAM2P method and baselines using Tensorflow (Abadi et al. 2016) (version 1.15) and Python (version 3.6). Other required packages are scikit-learn and numpy.

## Running source code samples
Here, we provide the instructions for using the source code samples of our DAM2P method the on the pair of the source domain (FFmpeg) to the target domain (LibPNG). For our method and baselines, after training the model on the training set, we can find out the best model (i.e., based on the results of AUC, Recall, Precision and F1-measure on the validation set) which then will be used to obtain the best results on the testing set. In our source code samples, for the demonstration purpose, we design to train the model using the training set, evaluate the model performance on the testing set after every specific iteration, and save the high results (in a list) on the testing set corresponding to the used hyper-parameters.

## Folders and files

The folder named “datasets” consists of all of the necessary files containing the source domain (FFmpeg) and target domain (LibPNG) data.  The file named “DAM2P_train_evaluate.py” is the source code of our proposed DAM2P method used for the training and evaluating processes. The file named “Utils.py” is a collection of supported Python functions used in the training and evaluating processes of the model.

## The model configuration 

For the DAM2P model configuration, please read the Model configuration section in the appendix of our paper for details.

## Citation

If you reference (and/or) use our source code samples in your work, please kindly cite our paper.

@misc{vannguyen-dam2p-2022,<br/>
  doi = {10.48550/ARXIV.2209.10406},<br/>
  url = {https://arxiv.org/abs/2209.10406},<br/>
  author = {Nguyen, Van and Le, Trung and Tantithamthavorn, Chakkrit and Grundy, John and Nguyen, Hung and Phung, Dinh},<br/>
  title = {Cross Project Software Vulnerability Detection via Domain Adaptation and Max-Margin Principle},<br/>
  publisher = {arXiv},<br/>
  year = {2022},<br/>
  copyright = {Creative Commons Attribution 4.0 International}<br/>
}
