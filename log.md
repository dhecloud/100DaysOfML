# 100 Days Of ML Code

### Day 0: 7 June 2018

**Today's Progress**:  
Hand Pose: Worked on understanding dlib's extract layer and wrote some testing code to figure out how offset in the parameters affected the output.

**Thoughts:**  
Hand Pose: Took longer than expected for this portion of the code due to the lack of familiarity of c++ and help from dlib's community (unlike python's pytorch/tensorflow) Hope to get this done ASAP.

**Link to work:** [Commit](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/b982bb94a330a486e6ae90e208f7c562e1b4c966)

### Day 1: 8 June 2018

**Today's Progress**:  
Hand Pose: Managed to implement dlib's extract layer. However i am changing from dlib to caffe for my implementation so will stop on implementing dlib. Spent some time installing caffe.  
DotA_Predictor: Revived my old DotA predictor project since i think i can make it better. 

**Thoughts:**  
Hand Pose: Even though past weeks of efforts studying dlib were wasted, pretty happy that i get to go back to python. Also wish building libraries on windows was easier..  
DotA_Predictor: Quite surprised by the amount of resources available for getting data for DotA.

**Link to work:** [(very messy Hand Pose Commit)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/4c152873f2d90f28a1d33eccc5e180f9d0b9bab2) [(DotA Commit)](https://github.com/dhecloud/DotA_Predictor/commit/ce31ebee589d82fa15163b03f0a2b990eb2f2629)

### Day 2: 9 June 2018

**Today's Progress**:  
Hand Pose: Gave up on installing pycaffe. will be implementing code in pytorch for hand pose  
DotA_Predictor: Added data scraping code for open dota for dota predictor project. 

**Thoughts:**  
Hand Pose: Still wish building on windows was easier.. seem to run into errors at every command.  
DotA_Predictor: Going to collect about 30k match data before i start retraining the predictor and adding some features.

**Link to work:** [(very messy Hand Pose Commit 2)](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/f637fdac8665d6add3281fb01ae36f13f306815e) [(DotA Commit)](https://github.com/dhecloud/DotA_Predictor/commit/1b7aa565069ed952e995827a223bb37656c4806e)

### Day 3: 10 June 2018

**Today's Progress**:  
Hand Pose: Added preliminary code for data loading and the network code for Region Ensemble Network.  
DotA_Predictor: Didnt do much work on it today.

**Thoughts:**  
Hand Pose: Pytorch is a breeze but i still have some confusion around the dimensions of the tensors. Need to clarify and clean up  
DotA_Predictor: -

**Link to work:** [Hand Pose](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/6d77a0025448415fd4e6fecd445408a3cdf8f581)

### Day 4: 11 June 2018

**Today's Progress**:  
Hand Pose: Added intermediate code for training and testing. Dataloading and REN code completed.  
DotA_Predictor: Didnt do much work on it today.

**Thoughts:**  
Hand Pose: Not much, happy to be working with pytorch. Hope my implementation of the REN and data loading is accurate. will probably need to refactor.  
DotA_Predictor: -

**Link to work:** [Hand Pose](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/15fd004811b16f566e8c663a3db37557e762b378)

### Day 5: 12 June 2018

**Today's Progress**:  
Hand Pose: Solving exploding gradient problem and double checked the network architecture. added normalization code.
DotA_Predictor: Didnt do much work on it today.

**Thoughts:**  
Hand Pose: For a while i could not figure out the reason for the exploding gradient even after double checking the data and the network arch. Turned out i had to use a more stable loss layer (SmoothL1Loss over MSE)  
DotA_Predictor: -

**Link to work:** [Hand Pose](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/86e66dd26d575c648c3b52f7929e8c6fac8bb8e1)

### Day 6: 13 June 2018

**Today's Progress**:  
Hand Pose: Added random sampling code. Spent some time trying to figure how to augment depth images. Tuned hyper params.  
DotA_Predictor: Didnt do much work on it today. 

**Thoughts:**  
Hand Pose: Work on depth images isnt as widespread as RGB. Hope i can figure out how to apply translations/scalings/rotations on normalized data.  
DotA_Predictor: -

**Link to work:** [Hand Pose](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/4c111778ed2f1ae0dcc04dc7d6d5194a6cb69c56)

### Day 7: 14 June 2018

**Today's Progress**:  
Hand Pose: Added testing code for drawing pose. Still figuring out how to do the augmentation.   
DotA_Predictor: Didnt do much work on it today. 

**Thoughts:**  
Hand Pose: Training is going pretty well, but the accuracy is limited by the lack of augmentation of the depth images.  
DotA_Predictor: -

**Link to work:** [Hand Pose](https://github.com/dhecloud/Hand-Pose-for-Rheumatoid-Arthritis/commit/e8e73a714248f3c2104b7577cb132ddff3fee856)