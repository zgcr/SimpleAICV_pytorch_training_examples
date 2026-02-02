python 13.0.inference_sam_single_image.py \
    --input-image-path '/root/code/SimpleAICV_pytorch_training_examples/gradio_demo/test_sam_images/truck.jpg' \
    --output-image-path '/root/code/SimpleAICV_pytorch_training_examples/inference_demo/inference_sam_result.jpg' \
    --input-prompt-points '[[500,375,1],[1125,625,1]]' \
    --input-prompt-box '[75, 275, 1725, 850]' \
    --input-prompt-mask '/root/code/SimpleAICV_pytorch_training_examples/gradio_demo/test_sam_images/truck_prompt_mask.jpg' \
    --mask-out-idx 0