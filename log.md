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
