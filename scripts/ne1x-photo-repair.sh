#!/bin/sh

python3.4 enhance.py \
    --train "$OPEN_IMAGES_PATH/*/*.jpg" --type photo --model repair \
    --epochs=50 --batch-shape=256 --device=gpu1 \
    --generator-downscale=2 --generator-upscale=2 \
    --generator-blocks=8 --generator-filters=128 --generator-residual=0 \
    --perceptual-layer=conv2_2 --smoothness-weight=1e7 --adversary-weight=0.0 \
    --train-noise=2.0 --train-jpeg=30

python3.4 enhance.py \
    --train "$OPEN_IMAGES_PATH/*/*.jpg" --type photo --model repair \
    --epochs=500 --batch-shape=240 --device=gpu1 \
    --generator-downscale=2 --generator-upscale=2 \
    --perceptual-layer=conv5_2 --smoothness-weight=5e3 --adversary-weight=5e1 \
    --generator-start=10 --discriminator-start=0 --adversarial-start=10 \
    --discriminator-size=48 \
    --train-noise=2.0 --train-jpeg=30
