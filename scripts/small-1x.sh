#!/bin/sh

python3.4 enhance.py \
    --train "$OPEN_IMAGES_PATH/*/*.jpg" --model small \
    --epochs=50 --batch-shape=192 --device=gpu0 \
    --generator-downscale=2 --generator-upscale=2 \
    --generator-blocks=8 --generator-filters=64 \
    --perceptual-layer=conv2_2 --smoothness-weight=1e7 --adversary-weight=0.0 \
    --train-blur=3 --train-noise=5.0

python3.4 enhance.py \
    --train "$OPEN_IMAGES_PATH/*/*.jpg" --model small \
    --epochs=500 --batch-shape=192 --device=gpu0 \
    --generator-downscale=2 --generator-upscale=2 \
    --perceptual-layer=conv5_2 --smoothness-weight=2e4 --adversary-weight=2e2 \
    --generator-start=5 --discriminator-start=0 --adversarial-start=5 \
    --discriminator-size=32 \
    --train-blur=3 --train-noise=5.0
