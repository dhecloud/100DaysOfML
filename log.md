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