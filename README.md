# CNN Architecture Transfer Study

Empirical comparison of five CNN configurations on the CIFAR-100 100-class image classification benchmark. The study examines the effect of architecture choice, ImageNet pretraining versus training from scratch, and CutMix data augmentation, evaluating each configuration in terms of top-1 accuracy, trainable parameter count, model size, and inference latency.

## Topics Covered

**Dataset and Preprocessing**
CIFAR-100 is downloaded automatically via torchvision. Images are upsampled to 224 × 224 pixels to match the input size expected by ImageNet-pretrained backbones. Training images are augmented with random resized cropping, random horizontal flipping, and color jitter. Test images are resized proportionally and center-cropped. Both splits are normalized using standard ImageNet channel statistics.

**Experimental Configurations**
Five configurations are implemented and compared:

- *ResNet-18 from scratch:* Trained entirely on CIFAR-100 with random initialization. Serves as a baseline to isolate the contribution of ImageNet pretraining.
- *ResNet-18 pretrained:* Initialized with ImageNet-1k weights. The original classification head is replaced with a new linear layer for 100 classes, and all parameters are fine-tuned.
- *ResNet-50 pretrained:* A deeper backbone fine-tuned with the same protocol as ResNet-18 pretrained, testing whether additional capacity improves accuracy.
- *MobileNetV2 pretrained:* A lightweight architecture designed for efficiency, used to evaluate the accuracy-efficiency trade-off against the ResNet family.
- *ResNet-18 pretrained + CutMix:* Extends the pretrained ResNet-18 with CutMix regularization. During training, a rectangular patch from a shuffled image replaces the corresponding region in each image, and the loss is computed as a weighted combination of cross-entropy for both the original and shuffled labels.

**Training Setup**
All configurations use SGD with Nesterov momentum and a cosine annealing learning rate schedule. Pretrained configurations use a learning rate of 0.001; the from-scratch configuration uses 0.01. All configurations train for 50 epochs with a batch size of 128 and a fixed random seed. Per-epoch train loss, train accuracy, test accuracy, learning rate, and wall-clock time are saved to `results/` as both JSON and CSV.

**Evaluation and Visualization**
After training, an evaluation script aggregates all five results and generates three figures: a test accuracy bar chart, a trainable parameter count versus accuracy scatter plot, and a model size comparison bar chart. A summary table is printed to stdout with each experiment's best accuracy, parameter count, model size, and inference latency.

**Data Efficiency Ablation**
All five configurations are retrained on 10%, 25%, and 100% of the CIFAR-100 training set for 10 epochs each using random subsampling. The resulting learning curves show how quickly each configuration reaches competitive accuracy as labeled data increases.
