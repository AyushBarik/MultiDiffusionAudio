---
description: Long-form audio generation development plan for AudioLDM
---

# AudioLDM Long-Form Audio Generation Plan

## Implementation Phases

### Phase 1: 3-Minute Audio Target (Current Focus)
- **Diffusion Model**: Implement MultiDiffusion for UNet
- **VAE Decoder**: Implement Tiled VAE for decoder
- **Vocoder**: Skip for now (HiFiGAN can handle ~4 minutes)

### Phase 2: Full Pipeline (Future)
- **Vocoder**: Implement chunked processing for HiFiGAN
- **Integration**: End-to-end long-form generation

## Technical Bottlenecks Identified

### 1. Diffusion Model Bottleneck
- **Problem**: Cannot handle 15,360 time steps for 10-minute audio
- **Root Cause**: Quadratic scaling in attention mechanisms
- **Solution**: MultiDiffusion - process chunks during each denoising step

### 2. VAE Decoder Bottleneck  
- **Problem**: Memory explosion with large feature maps
- **Root Cause**: Large tensor sizes in decoder operations
- **Solution**: Tiled VAE - decompose operations with overlapping tiles

### 3. Vocoder Bottleneck (Phase 2)
- **Problem**: HiFiGAN cannot handle very long spectrograms
- **Limit**: ~4 minutes maximum
- **Solution**: Chunked processing with overlap

## AudioLDM 20-Second Limit Context

### Root Cause (lines 170-172 in AudioLDM/audioldm/pipeline.py)
- Training distribution mismatch
- Model never trained on sequences longer than 10-20 seconds
- Results in NaN outputs due to:
  - Attention numerical instability
  - Batch normalization breakdown
  - Gradient flow issues

### Why MultiDiffusion + Tiled VAE Solves This
- **MultiDiffusion**: Keeps individual chunks within trained distribution
- **Tiled VAE**: Prevents memory explosion while maintaining quality
- **Global Coherence**: Overlap and merging strategies maintain consistency

## Implementation Priority Order

1. **MultiDiffusion for Diffusion UNet** (Highest Priority)
   - Focus on chunk processing during denoising steps
   - Implement proper overlap handling
   - Test with 3-minute audio targets

2. **Tiled VAE for VAE Decoder** (High Priority)
   - Implement tile decomposition
   - Share GroupNorm statistics
   - Ensure seamless merging
