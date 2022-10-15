python detect_single_image.py \
    --seed 0 \
    --model resnet50_fcos \
    --decoder FCOSDecoder \
    --trained_dataset_name COCO \
    --trained_num_classes 80 \
    --input_image_size 800 \
    --image_resize_style retinastyle \
    --min_score_threshold 0.5 \
    --trained_model_path /root/code/SimpleAICV-ImageNet-CIFAR-COCO-VOC-training/pretrained_models/detection_training/fcos/resnet50_fcos-coco-retinaresize800-metric37.864.pth \
    --test_image_path /root/code/SimpleAICV-ImageNet-CIFAR-COCO-VOC-training/inference_examples/resources/images/000000001551.jpg \
    --save_image_path /root/code/SimpleAICV-ImageNet-CIFAR-COCO-VOC-training/inference_examples \
    # --show_image