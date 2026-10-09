# CPU verification only

15 automated tests passed. Registered workflow completed: develop -> freeze (exploratory) -> finalize. Two tiny smoke models, one seed, two epochs, four training images and three evaluation images. AP=0 is a software check, not a research finding. The single evaluation group cannot support group-bootstrap confidence intervals.

YOLO and RT-DETRv2 adapters passed offline forward/backward/inference/reload tests with reduced test configurations. No RTX4090/CUDA training, full pretrained benchmark, or real external-data evaluation has been performed. CPU verification used torch 2.14.0+cpu / torchvision 0.29.0+cpu; Ubuntu setup pins torch 2.7.1 / torchvision 0.22.1.

Source grouping metadata remains unverified. The two cross-split similarity candidates were visually reviewed as different scenes; this does not establish independent capture sessions. The frozen smoke protocol contains original workspace paths and is historical verification evidence, not a protocol to reuse after moving the project. Create a new protocol for real runs.
