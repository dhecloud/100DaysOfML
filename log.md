# 100 Days Of ML Code

### Day 0: 7 July 2018

**Today's Progress**:  
Hand Pose: Worked on understanding dlib's extract layer and wrote some testing code to figure out how offset in the parameters affected the output.

**Thoughts:**  
Hand Pose: Took longer than expected for this portion of the code due to the lack of familiarity of c++ and help from dlib's community (unlike python's pytorch/tensorflow) Hope to get this done ASAP.

**Link to work:** [Commit](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/b982bb94a330a486e6ae90e208f7c562e1b4c966)

### Day 1: 8 July 2018

**Today's Progress**:  
Hand Pose: Managed to implement dlib's extract layer. However i am changing from dlib to caffe for my implementation so will stop on implementing dlib. Spent some time installing caffe.  
DotA_Predictor: Revived my old DotA predictor project since i think i can make it better.

**Thoughts:**  
Hand Pose: Even though past weeks of efforts studying dlib were wasted, pretty happy that i get to go back to python. Also wish building libraries on windows was easier..  
DotA_Predictor: Quite surprised by the amount of resources available for getting data for DotA.

**Link to work:** [(very messy Hand Pose Commit)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/4c152873f2d90f28a1d33eccc5e180f9d0b9bab2) [(DotA Commit)](https://github.com/dhecloud/DotA_Predictor/commit/ce31ebee589d82fa15163b03f0a2b990eb2f2629)

### Day 2: 9 July 2018

**Today's Progress**:  
Hand Pose: Gave up on installing pycaffe. will be implementing code in pytorch for hand pose  
DotA_Predictor: Added data scraping code for open dota for dota predictor project.

**Thoughts:**  
Hand Pose: Still wish building on windows was easier.. seem to run into errors at every command.  
DotA_Predictor: Going to collect about 30k match data before i start retraining the predictor and adding some features.

**Link to work:** [(very messy Hand Pose Commit 2)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/f637fdac8665d6add3281fb01ae36f13f306815e) [(DotA Commit)](https://github.com/dhecloud/DotA_Predictor/commit/1b7aa565069ed952e995827a223bb37656c4806e)

### Day 3: 10 July 2018

**Today's Progress**:  
Hand Pose: Added preliminary code for data loading and the network code for Region Ensemble Network.  
DotA_Predictor: Didnt do much work on it today.

**Thoughts:**  
Hand Pose: Pytorch is a breeze but i still have some confusion around the dimensions of the tensors. Need to clarify and clean up  
DotA_Predictor: -

**Link to work:** [Hand Pose](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/6d77a0025448415fd4e6fecd445408a3cdf8f581)

### Day 4: 11 July 2018

**Today's Progress**:  
Hand Pose: Added intermediate code for training and testing. Dataloading and REN code completed.  
DotA_Predictor: Didnt do much work on it today.

**Thoughts:**  
Hand Pose: Not much, happy to be working with pytorch. Hope my implementation of the REN and data loading is accurate. will probably need to refactor.  
DotA_Predictor: -

**Link to work:** [Hand Pose](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/15fd004811b16f566e8c663a3db37557e762b378)

### Day 5: 12 July 2018

**Today's Progress**:  
Hand Pose: Solving exploding gradient problem and double checked the network architecture. added normalization code.
DotA_Predictor: Didnt do much work on it today.

**Thoughts:**  
Hand Pose: For a while i could not figure out the reason for the exploding gradient even after double checking the data and the network arch. Turned out i had to use a more stable loss layer (SmoothL1Loss over MSE)  
DotA_Predictor: -

**Link to work:** [Hand Pose](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/86e66dd26d575c648c3b52f7929e8c6fac8bb8e1)

### Day 6: 13 July 2018

**Today's Progress**:  
Hand Pose: Added random sampling code. Spent some time trying to figure how to augment depth images. Tuned hyper params.  
DotA_Predictor: Didnt do much work on it today.

**Thoughts:**  
Hand Pose: Work on depth images isnt as widespread as RGB. Hope i can figure out how to apply translations/scalings/rotations on normalized data.  
DotA_Predictor: -

**Link to work:** [Hand Pose](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/4c111778ed2f1ae0dcc04dc7d6d5194a6cb69c56)

### Day 7: 14 July 2018

**Today's Progress**:  
Hand Pose: Added testing code for drawing pose. Still figuring out how to do the augmentation.   
DotA_Predictor: Didnt do much work on it today.

**Thoughts:**  
Hand Pose: Training is going pretty well, but the accuracy is limited by the lack of augmentation of the depth images.  
DotA_Predictor: -

**Link to work:** [Hand Pose](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/e8e73a714248f3c2104b7577cb132ddff3fee856)

### Day 8: 15 July 2018

**Today's Progress**:  
Hand Pose: Added preliminary code for data augmentation.  
DotA_Predictor: Didnt do much work on it today.

**Thoughts:**  
Hand Pose: Augmentation is taking pretty long. Need to figure out how to speed it up.  
DotA_Predictor: -

**Link to work:** [Hand Pose](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/f4d2f82532593bb78b15f92cd1172a36f9625c0f)

### Day 9: 16 July 2018

**Today's Progress**:  
Hand Pose: Data augmentation into h5py done, changing (x,y,z) into (u,v,d) ground truth also done.  
DotA_Predictor: Didnt do much work on it today.

**Thoughts:**  
Hand Pose: TIL a few things. 1. That x,y,z were not interchangable, and 2. Not to load all the augmentated images into memory. Need to redo data loading code to load from h5py or possibly just refactor to load from txt.  
DotA_Predictor: -

**Link to work:** [Hand Pose](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/1cf5635140af709b0a3207619366a84d224ffd0d)

### Day 10: 17 July 2018

**Today's Progress**:  
Hand Pose: rewrote `__get_item__` for MSRA dataset class. Commenced training but will need to find a better GPU.  
DotA_Predictor: Started filtering all pick matches out of the whole list of matches.

**Thoughts:**  
Hand Pose: Getting into the flow of code writing seems easier now. Training is currently limited by the 2GB memory GPU which reduces the batch size.. otherwise it could be faster.  
DotA_Predictor: -

**Link to work:** [(Hand Pose)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/76927e38e03abad3496c962be803c242a9d8b1e0) [(DotA Commit)](https://github.com/dhecloud/DotA_Predictor/commit/fbeb0acf19079c6cf5b9baf367c721cc8ec32fc5)

### Day 11: 18 July 2018

**Today's Progress**:  
Hand Pose: Debugged a big mistake on augmenting ground truth joints    
DotA_Predictor: Didnt do much work on it today.  
Paper: Started reading the paper on NASNet

**Thoughts:**  
Hand Pose: Training for 18 hours kinda wasted because the augmentation data was wrong :( The loss however went down. Should include more error metrics to check while training  
DotA_Predictor: -  
Paper: Interesting! Cant wait to finish reading it

**Link to work:** [(Hand Pose)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/ca39b830f707ee2c3e55e50c1c5e1f34e919008c) [(Paper)](https://arxiv.org/pdf/1707.07012.pdf)

### Day 12: 19 July 2018

**Today's Progress**:  
Hand Pose: fixed joints again (flipped horizontal coordinates)
DotA_Predictor: Didnt do much work on it today.  
Paper: Finished reading the paper on NASNet

**Thoughts:**  
Hand Pose: Training results are pretty good, probably need to train for a longer time
DotA_Predictor: -  
Paper: -

**Link to work:** [(Hand Pose)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/920a8f92cf56d200c4604f7babfed6b18d72c79a) [(Paper)](https://arxiv.org/pdf/1707.07012.pdf)

### Day 13: 20 July 2018

**Today's Progress**:  
Hand Pose: Did a little troubleshooting after training for an 18 hours. Training seems well except that more epoches need to be done
DotA_Predictor: Didnt do much work on it today.  
Paper: Started reading paper on Deep Pose: Human Pose Estimation

**Thoughts:**  
Hand Pose: Training results are pretty good, probably need to train for a longer time again
DotA_Predictor: -  
Paper: -

**Link to work:** [Paper](https://arxiv.org/pdf/1312.4659.pdf)

### Day 14: 21 July 2018

**Today's Progress**:  
Hand Pose: More training
DotA_Predictor: Didnt do much work on it today.  
Paper: Finished reading paper on Deep Pose: Human Pose Estimation

**Thoughts:**  
Hand Pose: -
DotA_Predictor: -  
Paper: DeepPose is a generic CNN applied on human pose in 2013. Probably replaced by better architectures.

**Link to work:** [Paper](https://arxiv.org/pdf/1312.4659.pdf)

### Day 15: 22 July 2018

**Today's Progress**:  
Hand Pose: Checked testing and validation progress and more training  
DotA_Predictor: Didnt do much work on it today.  
Paper: Started and finish reading paper on Real-time Articulated Hand Pose Estimation using Semi-supervised Transductive Regression Forests

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: -  
Paper: One of the earlier research on hand pose before CNNs. Random forest is something i'm not familiar with, need to do more reading.  

**Link to work:** [Paper](http://openaccess.thecvf.com/content_iccv_2013/papers/Tang_Real-Time_Articulated_Hand_2013_ICCV_paper.pdf)

### Day 16: 23 July 2018

**Today's Progress**:  
Hand Pose: Checked testing and validation progress and more training  
DotA_Predictor: Didnt do much work on it today.  
Paper: Started and finish reading paper on Efficient Hand Pose Estimation from a Single Depth Image

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: -  
Paper: One of the earlier research on hand pose before CNNs. Also a very basic random forest approach

**Link to work:** [Paper](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Xu_Efficient_Hand_Pose_2013_ICCV_paper.pdf)

### Day 17: 24 July 2018

**Today's Progress**:  
Hand Pose: Increased learning rate  
DotA_Predictor: Didnt do much work on it today.  
Paper: -

**Thoughts:**  
Hand Pose: Think learning rate has been decaying too fast, reduced the learning rate drop.  
DotA_Predictor: -  
Paper: -

**Link to work:** -

### Day 18: 25 July 2018

**Today's Progress**:  
Hand Pose: increased training dataset size  
DotA_Predictor: Didnt do much work on it today.  
Paper: Started reading DensePose, a SOTA human pose estimation released by facebook

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: -  
Paper: So far there has a lot of work done on making a good dataset (densepose-coco). I guess having a good dataset is already half the battle won

**Link to work:** [Paper](https://arxiv.org/pdf/1802.00434.pdf)

### Day 19: 26 July 2018

**Today's Progress**:  
Hand Pose: continue training  
DotA_Predictor: Did some work to continue crawling data and add some features  
Paper: Finished reading DensePose, a SOTA human pose estimation released by facebook  

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: Reason why i been procrasinating seems to be that data collection is always very tedious.. just have to get it over and done with :(  
Paper: Seems to be a mixture of all the techniques used for body pose estimation so far: cascading, RPN, and a novel technique which removes background

**Link to work:** [(Paper)](https://arxiv.org/pdf/1802.00434.pdf) [(Commit)](https://github.com/dhecloud/DotA_Predictor/commit/53273cdc95145948a4a25f4786975b8c5f0757cb)

### Day 20: 27 July 2018

**Today's Progress**:  
Hand Pose: continue training  
DotA_Predictor: Added more features for gold and xp dependency
Paper: No papers today

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: Data done and ready to be collected. Now to think of meaningful features. I understand the data and the gameplay pretty well so i should be able to provide a deeper analysis
Paper: -

**Link to work:** [Commit](https://github.com/dhecloud/DotA_Predictor/commit/6946aa4b547ca21cccd38264170341ffd4ae9ef3)

### Day 21: 28 July 2018

**Today's Progress**:  
Hand Pose: continue training  
DotA_Predictor: Finished adding code for training, testing and data collection
Paper: No papers today

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: Training and testing accuracy currently stands at 70%, which is 10% higher than what i got previously. Also, I also do think my code is now more robust than the previous iteration. yay!  
Paper: -  

**Link to work:** [Commit](https://github.com/dhecloud/DotA_Predictor/commit/9f1f41c828c5dad9bc5f99666089a54dd419996c)

### Day 22: 29 July 2018

**Today's Progress**:  
Hand Pose: continue training  
DotA_Predictor: Added 1 more feature  
Paper: Started reading a paper about GAN  

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: -  
Paper: GANs is a part of computer vision i have not really read much about. Will be exciting to read about different kinds of GANs beyond the basic one.

**Link to work:** [(Commit)](https://github.com/dhecloud/DotA_Predictor/commit/52f1f1c67796decd95cf3038ab3e86cd65d10185) [(Paper)](https://arxiv.org/pdf/1807.09251.pdf)

### Day 23: 30 July 2018

**Today's Progress**:  
Hand Pose: continue training  
DotA_Predictor: added team complexity feature
Paper: continued reading GANimation

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: -  
Paper: Reading is slowly because i had to read up on some new terms halfway. Also this is an unsupervised method which i am not familiar with

**Link to work:** [(Commit)](https://github.com/dhecloud/DotA_Predictor/commit/bb518270607acdf1970c6e0ecc9fc28baf1300e8) [(Paper)](https://arxiv.org/pdf/1807.09251.pdf)

### Day 24: 31 July 2018

**Today's Progress**:  
Hand Pose: Checked how to integrate it using pykinect  
DotA_Predictor: Collecting data mostly, about 16k now.  
Paper: Finished reading GANimation. Started reading Tacotron  

**Thoughts:**  
Hand Pose: Using pykinect seems easy enough, but i will have to go down to the lab where the kinect is to give it a try.  
DotA_Predictor: Seems to be mostly done - all that's left is collecting enough data to achieve a good enough generalization. Hopefully i can collect enough before the api limit runs out  
Paper: Unsupervised training is definitely a lot more work and uses more complicated metrics for loss and training. Will read the original GAN paper to give me a better foundation  

**Link to work:** [(Commit)](https://github.com/dhecloud/DotA_Predictor/commit/00945e5ad5dbb326187567105ebc9d9a4de9eadf) [(Paper 1)](https://arxiv.org/pdf/1807.09251.pdf) [(Paper 2)](https://arxiv.org/pdf/1703.10135.pdf)

### Day 25: 1 August 2018

**Today's Progress**:  
Hand Pose: -  
DotA_Predictor: Collecting data mostly, about 30k now  
Paper: Finished reading Tacotron.

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: Just constantly adding new features when i think of it  
Paper: Text-to-speech seems to be a pretty hard problem with various approaches  

**Link to work:** [(Paper)](https://arxiv.org/pdf/1703.10135.pdf)

### Day 26: 2 August 2018

**Today's Progress**:  
Hand Pose: -  
DotA_Predictor: Tried to implement a MLP from scratch in pytorch  
Paper: Finished reading GAN - the original paper

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: Implementing in pytorch was pretty fast, but training results are different from sklearn's wrapper. Probably got to do with learning rates  
Paper: Getting familiar with the basics  

**Link to work:** [(Commit)](https://github.com/dhecloud/DotA_Predictor/commit/cfe5f891d66dc0f79504091574a8072da4beeeba) [(Paper)](https://arxiv.org/pdf/1406.2661.pdf)

### Day 27: 3 August 2018

**Today's Progress**:  
Hand Pose: -  
DotA_Predictor: -  
Paper: Started reading StarGAN 

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: -  
Paper: StarGAN has a pretty novel architecture to deal with multi domain learning. I wonder if there are any other naturally occuring shapes or design that we can be inspired by

**Link to work:** [(Paper)](https://arxiv.org/pdf/1711.09020.pdf)

### Day 27: 4 August 2018

**Today's Progress**:  
Hand Pose: -  
DotA_Predictor: - 
Paper: Finished reading StarGAN 

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: -  
Paper: Thinking up new formulaes for loss functions requires a fair bit of intuition and mathematical background. Hope i can get to that stage one day

**Link to work:** [(Paper)](https://arxiv.org/pdf/1711.09020.pdf)

### Day 28: 5 August 2018

**Today's Progress**:  
Hand Pose: -  
DotA_Predictor: - 
Paper: Started and finished reading pix2pix on Image-to-Image Translation using cGANs

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: -  
Paper: GANs seem very promising in creating new content. I wonder if i can use speech data in GANs too

**Link to work:** [(Paper)](https://arxiv.org/pdf/1611.07004.pdf)

### Day 29: 6 August 2018

**Today's Progress**:  
Hand Pose: -  
DotA_Predictor: Reduced size of NN  
Paper: Started and finished reading Context encoders: inpainting

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: Reduced the size of the NN and to my surprise it performed better, albeit with lower training accuracy. This might be the case of occam razor, where a smaller NN might be a better one  
Paper: Inpainting is a niche area with pratical uses. GANs which i have been reading thus far seems to have similar approaches and learning lessons.

**Link to work:** [(Paper)](https://arxiv.org/pdf/1604.07379.pdf)

### Day 30: 7 August 2018

**Today's Progress**:  
Hand Pose: -  
DotA_Predictor: -  
Paper: Started and finished reading deep multi-scale video prediction

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: -  
Paper: This paper uses a more engineering approach in solving this problem. Evaluation metrics are different. Predicting the future in my opinion is a very hard task and will take a lot more research to fully solve  

**Link to work:** [(Paper)](https://arxiv.org/pdf/1511.05440.pdf)

### Day 31: 8 August 2018

**Today's Progress**:  
Hand Pose: -  
DotA_Predictor: -  
Paper: Started and finished reading Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: -  
Paper: super-resolution is a common practical application. With further research, it might be possible to even up-resolution old movies. I wonder what is the upper limit on up-resolutions. Can it go beyond 8K?

**Link to work:** [(Paper)](https://arxiv.org/pdf/1609.04802.pdf)

### Day 32: 9 August 2018

**Today's Progress**:  
Hand Pose: -  
DotA_Predictor: Tried fine-tuning hyper-parameters of xgboost  
Paper: Started and finished reading Unsupervised Image-to-Image Translation Networks  

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: Finetuning didnt really work. Data is capped currently too because of api limit. 73% on mlp is the best result so far.    
Paper: This paper was quite mathematical and slightly more theoretical than what i am used to. Quite surprised that i was able to understand most of it. Also learned some new terms.  

**Link to work:** [(Paper)](https://papers.nips.cc/paper/6672-unsupervised-image-to-image-translation-networks.pdf)

### Day 33: 10 August 2018

**Today's Progress**:  
Hand Pose: -  
DotA_Predictor: -  
Paper: Started reading Auto-Encoding Variational Bayes

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: -  
Paper: This paper is a pretty popular paper and served as the foundation of several SOTA GANs. It is also very mathematical and is pretty much testing my knowledge of stats and probability. Taking a longer time to read this. 

**Link to work:** [(Paper)](https://arxiv.org/pdf/1312.6114.pdf)

### Day 34: 11 August 2018

**Today's Progress**:  
Hand Pose: -  
DotA_Predictor: -  
Paper: Finished reading Auto-Encoding Variational Bayes

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: -  
Paper: Very mathematical. I gained a little insight behind the workings of VAEs, though i skimmed through some parts because it was getting too dry.

**Link to work:** [(Paper)](https://arxiv.org/pdf/1312.6114.pdf)

### Day 35: 12 August 2018

**Today's Progress**:  
Hand Pose: -  
DotA_Predictor: -  
Paper: Started reading DiscoGAN

**Thoughts:**  
Hand Pose: -  
DotA_Predictor: -  
Paper: -

**Link to work:** [(Paper)](https://arxiv.org/pdf/1703.05192.pdf)

### Day 36: 13 August 2018

**Today's Progress**:  
Hand Pose: Changed dataloading code to generate transformation on the fly  
DotA_Predictor: -  
Paper: Finished reading DiscoGAN

**Thoughts:**  
Hand Pose: Trying out the free gpu cluster is troublesome as i cannot save the data to the remote server. Though i guess doing data augmentation on the fly is much better.  
DotA_Predictor: -  
Paper: Didnt really seem ground-breaking to me. Just a simple case of more (GANs) is more

**Link to work:** [(Paper)](https://arxiv.org/pdf/1703.05192.pdf) [(Commit)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/f747a99d1bf6334082cfc453653368f23e53203d)

### Day 37: 14 August 2018

**Today's Progress**:  
Hand Pose: Changed some code to work on linux distros  
DotA_Predictor: -  
Paper: -  

**Thoughts:**  
Hand Pose: Spent most of the time today debugging and trying to get it to work on RedHat. Not really ML, but related i guess  
DotA_Predictor: -  
Paper: -

**Link to work:** [(Commit)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/06af49540a1e086bd9c57e96e4c2aa8a91d5d8d2)

### Day 38: 15 August 2018

**Today's Progress**:  
Hand Pose: Final update for linux distros. Also added full set of the dataset  
DotA_Predictor: -  
Paper: -  

**Thoughts:**  
Hand Pose: Finally got it to work on the HPC. Also added full dataset so that it can potentially give better results  
DotA_Predictor: Prolly done with this project for now until i get new ideas. Not including this project in future journals   
Paper: -

**Link to work:** [(Commit)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/e9516e4422bc89af12b9cce0824a91df0582f407)

### Day 39: 16 August 2018

**Today's Progress**:  
Hand Pose: -  
Paper: Started reading CycleGAN  

**Thoughts:**  
Hand Pose: -   
Paper: CycleGAN seems to be pretty similar to Disco GAN. i googled and seems like the differences are in the loss functions, but they are pretty minor. CycleGAN also has an addition hyperparameter to adjust contribution of reconstruction loss. 

**Link to work:** [(Paper)](https://arxiv.org/pdf/1703.10593.pdf)

### Day 40: 17 August 2018

**Today's Progress**:  
Hand Pose: Added code for augmenting data - rotation and scaling    
Paper: Finished reading CycleGAN  

**Thoughts:**  
Hand Pose: Initially thought that tracking landmarks for rotation and scaling would be extremely hard, but turned out easier than expected.  
Paper: -  

**Link to work:** [(Commit)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/6a35c949994695c6d9ea0ce13a7fbe7a113d9f23) [(Paper)](https://arxiv.org/pdf/1703.10593.pdf)

### Day 41: 18 August 2018

**Today's Progress**:  
Hand Pose: Did more training  
Paper: Played around with the code for CycleGAN    

**Thoughts:**  
Hand Pose: Thought increasing the batch size and data would help with the training, but strangely it did badly on the validation set and even on training data. Need to figure out what went wrong. Gonna tune hyperparameters for now.  
Paper: CycleGAN really needs a lot of GPU memory. I cannot even run it on my 940m GPU. Even with batch size =1, it takes up 3.5G RAM apparently.  

**Link to work:** -

### Day 42: 19 August 2018

**Today's Progress**:  
Hand Pose: cleanup round 1 
Paper: Played around with the code for CycleGAN    

**Thoughts:**  
Hand Pose: cleaning up my code made me realise the importance of going through different approaches to data loading. i tried saving to disk first, then into h5py files, then finally doing augmentation on the fly. Easy to take for granted how many iterations code have gone through before the final product
Paper: Going to try hand pose using cycleGAN which depth images translated to their joints. of course, the drawback is that we will not be able to get the joints coordinates directly.  

**Link to work:** [(Commit)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/3cb1d7255657df9f4e470208ce9051de9b425436)

### Day 43: 20 August 2018

**Today's Progress**:  
Hand Pose: major revamp for the network  
Paper: -   

**Thoughts:**  
Hand Pose: Realised a major flaw in my code which in hindsight was quite stupid. (layers cannot be reused!)  
Paper: -  

**Link to work:** [(Commit)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/7c116eb670c2794544915d6c936ca09692efd631)

### Day 44: 21 August 2018

**Today's Progress**:  
Hand Pose: cleanup round 2 and minor refactoring  
Paper: -   

**Thoughts:**  
Hand Pose: tried to refactor code into the argparse way that most popular repos seem to code in     
Paper: -  

**Link to work:** [(Commit)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/f7649a79d2c1ea0e49ac1e9703b79e43d3363652)


### Day 45: 22 August 2018

**Today's Progress**:  
Hand Pose: training troubleshooting  
Paper: -   

**Thoughts:**  
Hand Pose: Trying to figure out why training on a full dataset does not train well    
Paper: -  

**Link to work:** -

### Day 46: 23 August 2018

**Today's Progress**:  
Hand Pose: More training, and minor cleanup  
Paper: -   

**Thoughts:**  
Hand Pose: seems like when augmenting the data, it does not work as well. Maybe there is some problem with my augmentation    
Paper: -  

**Link to work:** -

### Day 47: 24 August 2018

**Today's Progress**:  
Hand Pose: normalized ground truths  
Paper: -   

**Thoughts:**  
Hand Pose: Problem with augmenting was that augmenting the image did not correspond to the same translation in the joints. had to translate the joints too before augmenting
Paper: -  

**Link to work:** [(Commit)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/4bd69ac97cbf782c54a98190850da766558e1d94)


### Day 48: 25 August 2018

**Today's Progress**:  
Hand Pose: Fixed normalization  
Paper: -   

**Thoughts:**  
Hand Pose: fixed normalization of joints. hope it goes well. cant seem to get the first iteration of my implementation right, which wastes a lot of time
Paper: -  

**Link to work:** [(Commit)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/160c7071eae56f528b9bd727a97ff8253913e9cf)

### Day 49: 26 August 2018

**Today's Progress**:  
Hand Pose: More debugging, added minor features  
Paper: -   

**Thoughts:**  
Hand Pose: Cant figure out why in the world is the training not going well  
Paper: -  

**Link to work:** [(Commit)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/a9fcedd027d740561f9765558423ab2a143462d7)

### Day 50: 27 August 2018

**Today's Progress**:  
Hand Pose: Added xavier initialization and custom smooth l1 loss according to the paper  
Paper: -   

**Thoughts:**  
Hand Pose: Cant figure out why in the world is the training not going well. Reached out to stack overflow and forums for help :(  
Paper: -  

**Link to work:** [(Commit)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/ba83cb1ef0af6bc6eca3acffcf02ad7b7a89a3a8)

### Day 51: 28 August 2018

**Today's Progress**:  
Hand Pose: Solved problem. Fully working code  
Kaggle: Saw this competition on detecting pneumonia from radiographs.  
Paper: -   

**Thoughts:**  
Hand Pose: Turned out error was really silly. indexing tensors in pytorch are called by reference and new variables are not created.  
Kaggle: Might focus on this project once the hand pose project is concluded  
Paper: -  

**Link to work:** [(Commit)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/d7bce10e2f691a32f0996b669b07324ee336c62c) [(Kaggle)](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

### Day 52: 29 August 2018

**Today's Progress**:  
Hand Pose: -    
Kaggle: Checked out the dcm files and read up a bit on rcnn  
Paper: -   

**Thoughts:**  
Hand Pose: Training is going fine. Augmentation is kind of making the results weird though.  
Kaggle: Have a few ideas on how to approach this, but i will probably try out the faster-rcnn architecture on it first.  
Paper: -  

**Link to work:** [(Hand Pose)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/8276444e80d2aa0fa2570212ebf075d70ec44c44) [(Kaggle)](https://github.com/dhecloud/Kaggle-Challenge/commit/5032cc39c497ae577612ad4a25a8b111238492d3)

### Day 53: 30 August 2018

**Today's Progress**:  
Hand Pose: -    
Kaggle: Coded a little more, using resnet as starter code  
Paper: Started on RCNN paper  

**Thoughts:**  
Hand Pose: -   
Kaggle: Posing the problem as a classification problem first, and see if the postive/negative images can be distinguished  
Paper: wanted to revise on selective search and the fast rcnn architecture  

**Link to work:** [(Paper)](https://arxiv.org/pdf/1311.2524.pdf) [(Kaggle)](https://github.com/dhecloud/Kaggle-Challenge/commit/13d32893e59d1b93659e3c1520fdf6170d26ac1a)

### Day 54: 31 August 2018

**Today's Progress**:  
Hand Pose: -    
Kaggle: Tried adapting rcnn and unet  
Paper: -  

**Thoughts:**  
Hand Pose: -   
Kaggle: Trying to adapt faster-rcnn and unet and see if it works  
Paper: -

**Link to work:** [(Kaggle)](https://github.com/dhecloud/Kaggle-Challenge/commit/10e6cfcfe4623882e9af603bd44dfd2854e8f30f)

### Day 55: 1 September 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: More adaptions to code  
Paper: Almost finished RCNN paper  

**Thoughts:**  
Hand Pose: -  
Kaggle: trying to get the code working on the server    
Paper: Main body of the paper did not feel very insightful. Appendix seems to contain more intuition

**Link to work:** [(Paper)](https://arxiv.org/pdf/1311.2524.pdf) [(Kaggle)](https://github.com/dhecloud/Kaggle-Challenge/commit/9139ae9a9f16031dacad46694d1192c2c263ba7a)

### Day 56: 2 September 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: -  
Paper: Finished RCNN paper  

**Thoughts:**  
Hand Pose: -  
Kaggle: -    
Paper: -  

**Link to work:** [(Paper)](https://arxiv.org/pdf/1311.2524.pdf)

### Day 57: 3 September 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: Spent some time trying to debug  
Paper: -  

**Thoughts:**  
Hand Pose: -  
Kaggle: Tried to build pytorch from scratch.. running into some cuda errors now T.T sucks to have no sudo access
Paper: -  

**Link to work:** -  

### Day 58: 4 September 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: Spent some time trying to debug  
Paper: -  

**Thoughts:**  
Hand Pose: -  
Kaggle: Just another day of doing the same thing
Paper: -  

**Link to work:** -  

### Day 59: 5 September 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: Spent some time trying to debug  
Paper: Started reading selective search paper    

**Thoughts:**  
Hand Pose: -  
Kaggle: Just another day of doing the same thing
Paper: Selective search surprisingly turned out to be a graph problem

**Link to work:** [(Paper)](https://staff.fnwi.uva.nl/th.gevers/pub/GeversIJCV2013.pdf)

### Day 60: 6 September 2018

**Today's Progress**:  
Hand Pose: Tried to extract the hand depth image from the kinect  
Kaggle: -  
Paper: -

**Thoughts:**  
Hand Pose: -   
Kaggle: -  
Paper: -  

**Link to work:** -  

### Day 61: 7 September 2018

**Today's Progress**:  
Hand Pose: Tried to build pytorch from source  
Kaggle: -  
Paper: Continued reading Selective Search

**Thoughts:**  
Hand Pose: -   
Kaggle: -  
Paper: -  

**Link to work:** [(Paper)](https://staff.fnwi.uva.nl/th.gevers/pub/GeversIJCV2013.pdf)

### Day 62: 8 September 2018

**Today's Progress**:  
Hand Pose: Started writing the report draft  
Kaggle: -  
Paper: -  

**Thoughts:**  
Hand Pose: -   
Kaggle: -  
Paper: -  

**Link to work:** -  

### Day 63: 9 September 2018

**Today's Progress**:  
Hand Pose: Wrote the report draft. Wrote the README. slight refactoring.
Kaggle: -  
Paper: -  

**Thoughts:**  
Hand Pose: -   
Kaggle: -  
Paper: -  

**Link to work:** [(Hand Pose)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/c7ced31b822caa917fdf92e14252a39967220ee4) 

### Day 64: 10 September 2018

**Today's Progress**:  
Hand Pose: redid some training  
Kaggle: Learnt and recreated a docker image  
Paper: -  

**Thoughts:**  
Hand Pose: -   
Kaggle: Docker tutorial has really improved   
Paper: -  

**Link to work:** - 

### Day 65: 11 September 2018

**Today's Progress**:  
Hand Pose: Continued writing paper, grew graphs
Kaggle: -  
Paper: -  

**Thoughts:**  
Hand Pose: -   
Kaggle: apparently NSCC does not allow docker, have to use singularity :(
Paper: -  

**Link to work:** - 

### Day 66: 12 September 2018

**Today's Progress**:  
Hand Pose: -
Kaggle: Read a little on singularity  
Paper: -  

**Thoughts:**  
Hand Pose: -   
Kaggle: -  
Paper: -  

**Link to work:** - 

### Day 67: 13 September 2018

**Today's Progress**:  
Hand Pose: separated plotting code  
Kaggle: Read a little on singularity  
Paper: -  

**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  

**Link to work:** [(Hand Pose)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/c60dc8a9984213388b9626ac2236f809180a248b) 

### Day 68: 14 September 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: Tried to made singularity work  
Paper: -  

**Thoughts:**  
Hand Pose: -  
Kaggle: By now i think i am pretty compentent in docker.. with all the conversion to singularity and what not   
Paper: -  

**Link to work:** -

### Day 69: 15 September 2018

**Today's Progress**:  
Hand Pose: Did a bit of brainstorming on how to improve the report  
Kaggle: -  
Paper: -  

**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  

**Link to work:** -

### Day 70: 16 September 2018

**Today's Progress**:  
Hand Pose: -
Kaggle: Read a bit of singularity documentation
Paper: -  

**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  

**Link to work:** -

### Day 71: 17 September 2018

**Today's Progress**:  
Hand Pose: Refactored some code and added some minor testing code for calculating MAE  
Kaggle: -  
Paper: -  

**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  

**Link to work:** [(Hand Pose)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/aeab938453961b2c993990dfc9ab62f13538071c) 

### Day 72: 18 September 2018

**Today's Progress**:  
Hand Pose: Continued writing report  
Kaggle: -  
Paper: -  

**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  

**Link to work:** -

### Day 73: 19 September 2018

**Today's Progress**:  
Hand Pose: Redid many experiments   
Kaggle: -  
Paper: -  

**Thoughts:**  
Hand Pose: Redid many experiments for formality and consistency sake. Made me realise that hyperparameters really did play a big part in improving the results  
Kaggle: -  
Paper: -  

**Link to work:** -

### Day 74: 20 September 2018

**Today's Progress**:  
Hand Pose: Did some kinect code     
Kaggle: -  
Paper: -  

**Thoughts:**  
Hand Pose: Not sure if it's the camera or.. but the kinect is giving me weird depth images :/  
Kaggle: -  
Paper: -  

**Link to work:** [(Hand Pose)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/e9060039e6cdee435100debc3a740f1fb9161a42) 

### Day 75: 21 September 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: Coded some MLPs in pytorch from scratch.  


**Thoughts:**  
Hand Pose:  
Kaggle: -  
Paper: -  
School: Nothing i haven't done before

**Link to work:** [(School)](https://github.com/dhecloud/CZ4042-Assignment-1/commit/83031499bde19fcc394b666fd0f38f24140202e0) 

### Day 76: 22 September 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: Finished up the assignment


**Thoughts:**  
Hand Pose:  
Kaggle: -  
Paper: -  
School: Proud of myself for finishing it in 2 days!

**Link to work:** [(School)](https://github.com/dhecloud/CZ4042-Assignment-1/commit/400227a3eda54d44ef5e56092e97a29555670444) 

### Day 77: 23 September 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: Finished up the assignment


**Thoughts:**  
Hand Pose:  
Kaggle: -  
Paper: -  
School: Redid some experiments and finalized everything

**Link to work:** [(School)](https://github.com/dhecloud/CZ4042-Assignment-1/commit/5f10bccad860f5e32826133542014951c8baaafc) 

### Day 78: 24 September 2018

**Today's Progress**:  
Hand Pose: Continued writing and improving report  
Kaggle: -  
Paper: -  
School: Caught some bugs i didnt foresee  


**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: -

**Link to work:** [(School)](https://github.com/dhecloud/CZ4042-Assignment-1/commit/6ec06446bb60530aebe59e8699c49cef41d3e83f) 

### Day 79: 25 September 2018

**Today's Progress**:  
Hand Pose: Continued writing and improving report  
Kaggle: -  
Paper: -  
School: redid a section for k-fold validation  


**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: k-fold validation is so tedious!

**Link to work:** [(School)](https://github.com/dhecloud/CZ4042-Assignment-1/commit/4921354c26f8c862bad9243dfd3fcfe5d309e67a) 

### Day 80: 26 September 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: redid training  


**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: -  

**Link to work:** -  

### Day 81: 27 September 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: finally managed to get containers to work  
Paper: -  
School: redid a section for k-fold validation. Wrote some specific plotting codes  


**Thoughts:**  
Hand Pose: -  
Kaggle: so happy that i can finally train!! even though its a little too late.   
Paper: -  
School: -  

**Link to work:** [(School)](https://github.com/dhecloud/CZ4042-Assignment-1/commit/7e1efe2bd0d0a2b7013074c4ee837ae3b0ddd471) 

### Day 82: 28 September 2018

**Today's Progress**:  
Hand Pose: Finished kinect code    
Kaggle: finally managed to get containers to work  
Paper: -  
School: -  


**Thoughts:**  
Hand Pose: somehow the trained checkpoints do not give good results?? thats kinda disappointing and annoying  
Kaggle: -     
Paper: -  
School: -  

**Link to work:** -  

### Day 83: 29 September 2018

**Today's Progress**:  
Hand Pose: Retrained all experiments    
Kaggle: -  
Paper: -  
School: -  


**Thoughts:**  
Hand Pose: Finally have a working prototype. Yay!   
Kaggle: -  
Paper: -  
School: -  

**Link to work:** -  

### Day 84: 30 September 2018

**Today's Progress**:  
Hand Pose: Finished fyp and report  
Kaggle: -  
Paper: -  
School: -  


**Thoughts:**  
Hand Pose: project is coming to a close. Yay!  
Kaggle: -  
Paper: -  
School: -  

**Link to work:** [(Hand Pose)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/fe8462d55e60aa3c111d9be40ea2e8e43813b22a) 

### Day 85: 1 October 2018

**Today's Progress**:  
Hand Pose: Touched up fyp  
Kaggle: -  
Paper: -  
School: Helped friend with some code    


**Thoughts:**  
Hand Pose: prof forwarded me an email about something about a potential A. yay!  
Kaggle: -  
Paper: -  
School: -  

**Link to work:** -  

### Day 86: 2 October 2018

**Today's Progress**:  
Hand Pose: Started doing the poster  
Kaggle: -  
Paper: -  
School: Did some deep learning tutorials  


**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: Manually computing the gradients and back prop really made me appreciate pytorch  

**Link to work:** -  

### Day 87: 3 October 2018

**Today's Progress**:  
Hand Pose:  
Kaggle: -  
Paper: -  
School: Helped a friend do some work.


**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: Learnt in detail about a lot of architectures like alexnet, squeezenet, densenet, which i heard of in passing but never really delved in deep.  

**Link to work:** -  

### Day 88: 4 October 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: Helped a friend do some work again.


**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: -    

**Link to work:** -  

### Day 89: 5 October 2018

**Today's Progress**:  
Hand Pose: Starting doing the demo video    
Kaggle: -  
Paper: -  
School: Studied a little on my neural network mod  


**Thoughts:**  
Hand Pose: this is harder than machine learning T.T  
Kaggle: -  
Paper: -  
School: -  

**Link to work:** -  

### Day 90: 6 October 2018

**Today's Progress**:  
Hand Pose: Further touchups on report    
Kaggle: -  
Paper: -  
School: -  


**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: -  

**Link to work:** -  

### Day 91: 7 October 2018

**Today's Progress**:  
Hand Pose: -     
Kaggle: -  
Paper: Completed Selective Search paper  
School: -  


**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: Havent had time to read this in a long while  
School: -  

**Link to work:** -  [(Paper)](https://staff.fnwi.uva.nl/th.gevers/pub/GeversIJCV2013.pdf)

### Day 92: 8 October 2018

**Today's Progress**:  
Hand Pose: Major cleanup of code  
Kaggle: -  
Paper: -    
School: -  


**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: -  

**Link to work:** - [(Hand Pose)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/3f25a831854bb44e16c9c98b42fd9311744ba13b) 

### Day 93: 9 October 2018

**Today's Progress**:  
Hand Pose: Changed more report stuff  
Kaggle: Tried running faster-rcnn again now that i can run it on nscc  
Paper: -    
School: -  


**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: Need to start reading on intern stuff soon..  
School: -  

**Link to work:** - 

### Day 94: 10 October 2018

**Today's Progress**:  
Hand Pose: Changed more report stuff  
Kaggle: - 
Paper: -    
School: -  


**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: -  

**Link to work:** - 

### Day 95: 11 October 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: - 
Paper: -    
School: Not much, but started revising the stanford materials again  


**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: -  

**Link to work:** - 

### Day 96: 12 October 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: - 
Paper: -    
School: rewatching lecture 1 and 2 of the stanford nlp videoes  


**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: Coming back to rewatch the materials again makes a lot more sense. If only ntu had subtitles for every lecture ;(  

**Link to work:** - 

### Day 97: 13 October 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: - 
Paper: -    
School: rewatching lecture 3 of the stanford nlp videoes  


**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: Gonna try doing the assignments in pytorch instead of tensorflow  

**Link to work:** - 

### Day 98: 14 October 2018

**Today's Progress**:  
Hand Pose: -  
Kaggle: - 
Paper: -    
School: Finished lecture 3 of the stanford nlp videoes  


**Thoughts:**  
Hand Pose: -  
Kaggle: -  
Paper: -  
School: -  

**Link to work:** - 