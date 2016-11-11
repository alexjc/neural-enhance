#!/bin/sh

python3.4 enhance.py \
    --train "$OPEN_IMAGES_PATH/*/*.jpg" --type photo --model default \
    --epochs=50 --batch-shape=256 --device=gpu0 \
    --generator-downscale=0 --generator-upscale=1 \
    --generator-blocks=8 --generator-filters=128 --generator-residual=0 \
    --perceptual-layer=conv2_2 --smoothness-weight=1e7 --adversary-weight=0.0 \
    --train-noise=1.0

python3.4 enhance.py \
    --train "$OPEN_IMAGES_PATH/*/*.jpg" --type photo --model default \
    --epochs=500 --batch-shape=240 --device=gpu0 \
    --generator-downscale=0 --generator-upscale=1 \
    --perceptual-layer=conv5_2 --smoothness-weight=5e3 --adversary-weight=5e1 \
    --generator-start=10 --discriminator-start=0 --adversarial-start=10 \
    --discriminator-size=64 \
    --train-noise=1.0
