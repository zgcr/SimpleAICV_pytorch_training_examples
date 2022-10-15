python classify_single_image.py \
    --seed 0 \
    --model resnet50 \
    --trained_num_classes 1000 \
    --input_image_size 224 \
    --trained_model_path /root/code/SimpleAICV-ImageNet-CIFAR-COCO-VOC-training/pretrained_models/classification_training/resnet/resnet50-acc76.264.pth \
    --test_image_path /root/code/SimpleAICV-ImageNet-CIFAR-COCO-VOC-training/inference_examples/resources/images/000000030079.jpg \
    --save_image_path /root/code/SimpleAICV-ImageNet-CIFAR-COCO-VOC-training/inference_examples 
    # --show_image