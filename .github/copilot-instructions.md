# AudioLDM AI Agent Instructions

Always present code edits as diffs so the user can review and verify changes before applying them. THIS IS EXTREMELY IMPORTANT. DO NOT DIRECTLY EDIT CODE. 

IF YOU DONT USE DIFFS MY CODEBASE GET'S MESSED UP.

## Project Overview
AudioLDM is a text-to-audio generation model based on Latent Diffusion that can create high-quality audio from text descriptions.

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

## Development Focus Areas

### Long-Form Audio Generation
The current focus is extending AudioLDM beyond its ~20-second limit using two key approaches:

#### 1. MultiDiffusion
- Process overlapping audio chunks during diffusion process
- Core algorithm in `.github/instructions/multidiffusion.instructions.md`
- Implementation requires chunk processing with 50-frame overlaps
- Every diffusion timestep requires merging overlapping regions

#### 2. Tiled VAE
- Handle large tensors by processing tiles with shared normalization statistics
- Core algorithm in `.github/instructions/tiled-vae.instructions.md` 
- Critical: GroupNorm statistics must be shared across tiles
- Implementation requires padding (11px for decoder, 32px for encoder)

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

## Key Instruction Files

For more detailed implementation guides, refer to these files:

1. **Main Instructions**: Overall project structure and components
2. **MultiDiffusion Algorithm**: Detailed implementation of chunked diffusion in `.github/instructions/multidiffusion.instructions.md`
3. **Tiled VAE Algorithm**: Memory-efficient VAE processing in `.github/instructions/tiled-vae.instructions.md`
4. **Long-Form Generation Plan**: Development roadmap in `.github/instructions/long-form-generation.instructions.md`

## Verification of Code Edits
- Always present code edits as diffs so the user can review and verify changes before applying them. 