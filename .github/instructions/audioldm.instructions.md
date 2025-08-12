---
description: Core instructions for working with AudioLDM
---

# AudioLDM AI Agent Instructions

## Project Overview
AudioLDM is a text-to-audio generation model based on Latent Diffusion that can create high-quality audio from text descriptions. The project is focused on both standard generation (limited to ~20 seconds) and developing long-form audio generation via MultiDiffusion and Tiled VAE approaches.

## Key Architecture Components

### Core Model Architecture
- **Latent Diffusion Model** (`audioldm/ldm.py`): Central component implementing the diffusion process
- **VAE** (`audioldm/variational_autoencoder/`): Encodes/decodes between audio and latent space
- **CLAP** (`audioldm/clap/`): Contrastive Language-Audio Pretraining model for text conditioning
- **HiFiGAN** (`audioldm/hifigan/`): Vocoder to convert spectrograms to waveforms

### Data Flow
1. Text prompt → CLAP encoder → text embeddings
2. Random noise → UNet (conditioned on text) → denoised latent
3. Latent → VAE decoder → mel spectrogram
4. Mel spectrogram → HiFiGAN → waveform audio

## Critical Developer Workflows

### Testing Text-to-Audio Generation
```python
# Basic generation
from audioldm import build_model, text_to_audio

model = build_model(model_name="audioldm-m-full")
waveform = text_to_audio(
    latent_diffusion=model,
    text="A piano playing a gentle melody",
    duration=10.0,
    seed=42
)
```

### Working with Notebooks
- Use `audioldm_inference.ipynb` as the primary development interface
- Run setup cells 1-3 to initialize the environment and model
- Standard development flow: generate samples → analyze → implement changes → test

### File Naming and Organization
- Generated audio stored in `output/generation/` with timestamp-prompt naming
- Use kebab-case for new files in `.cursor/rules/`

## Common Pitfalls and Limitations

### Memory Management
- Diffusion models require significant VRAM (4GB+ minimum)
- Watch for OOM errors in MultiDiffusion/Tiled VAE implementation
- When implementing tiling, prioritize correct GroupNorm statistic sharing

### Duration Handling
- Model fails with NaN beyond ~20 seconds due to training distribution
- The duration_to_latent_t_size() function in pipeline.py converts duration to latent size
- Must handle chunk overlaps carefully to maintain global audio coherence

## Integration Patterns

### External Dependencies
- The model downloads pre-trained weights automatically at first use
- CLAP model provides text embedding functionality
- HiFiGAN handles the final conversion from spectrogram to waveform

### Key Extension Points
- `pipeline.py`: Main workflow orchestration for generation
- `ldm.py`: Core diffusion model implementation
- Chunking implementation should integrate with `DDIMSampler` for timestep processing

## Project Roadmap
1. Current: Implement MultiDiffusion for 3-minute generation
2. Next: Implement Tiled VAE for memory-efficient processing
3. Future: Add chunked vocoder processing for 10+ minute generation
