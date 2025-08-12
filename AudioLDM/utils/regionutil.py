import os
import numpy as np
import torch
import time
from audioldm import build_model
from audioldm.latent_diffusion.ddim import DDIMSampler
from audioldm.pipeline import duration_to_latent_t_size
from tqdm import tqdm
import matplotlib.pyplot as plt

class RegionConfig:
    """Configuration for region-based MultiDif    #            lat                      
                   latent = a_prev.sqrt() * pred_x0 + dir_xt + noise
            
            pbar.update(1)
    
    return latent  pbar.update(1)
    
    return latent  pbar.update(1)
    
    return latentt efficiency - now it's simpler: 1 model eval per region per timestep
    print(f"🚀 CORRECT MULTIDIFFUSION: {actual_evals} model evaluations total")      pbar.update(1)
    
    # Report efficiency - chunk-based with active region optimization
    total_chunks = len(get_audio_views(total_frames, window_size=256, stride=64))
    max_possible_evals = total_chunks * len(all_embeddings) * len(time_sequence)
    efficiency = (1.0 - actual_evals / max_possible_evals) * 100
    
    print(f"🚀 CHUNK-BASED MULTIDIFFUSION: {actual_evals} model evaluations total")
    print(f"📊 Efficiency: {actual_evals}/{max_possible_evals} evaluations ({efficiency:.1f}% reduction)")
    print(f"⚡ Saved by only processing active regions per chunk!")
    
    return latentdate(1)
    
    # Report efficiency - chunk-based with active region optimization
    total_chunks = len(get_audio_views(total_frames, window_size=256, stride=64))
    max_possible_evals = total_chunks * len(all_embeddings) * len(time_sequence)
    efficiency = (1.0 - actual_evals / max_possible_evals) * 100
    
    print(f"🚀 CHUNK-BASED MULTIDIFFUSION: {actual_evals} model evaluations total")
    print(f"📊 Efficiency: {actual_evals}/{max_possible_evals} evaluations ({efficiency:.1f}% reduction)")
    print(f"⚡ Saved by only processing active regions per chunk!")
    
    return latentprev.sqrt() * pred_x0 + dir_xt + noise
            
            pbar.update(1)
    
    # Report efficiency - chunk-based with active region optimization
    total_chunks = len(get_audio_views(total_frames, window_size=256, stride=64))
    max_possible_evals = total_chunks * len(all_embeddings) * len(time_sequence)
    efficiency = (1.0 - actual_evals / max_possible_evals) * 100
    
    print(f"🚀 CHUNK-BASED MULTIDIFFUSION: {actual_evals} model evaluations total")
    print(f"📊 Efficiency: {actual_evals}/{max_possible_evals} evaluations ({efficiency:.1f}% reduction)")
    print(f"⚡ Saved by only processing active regions per chunk!")
    
    return latentficiency - chunk-based with active region optimization
    total_chunks = len(get_audio_views(total_frames, window_size=256, stride=64))
    max_possible_evals = total_chunks * len(all_embeddings) * len(time_sequence)
    efficiency = (1.0 - actual_evals / max_possible_evals) * 100
    
    print(f"🚀 CHUNK-BASED MULTIDIFFUSION: {actual_evals} model evaluations total")
    print(f"📊 Efficiency: {actual_evals}/{max_possible_evals} evaluations ({efficiency:.1f}% reduction)")
    print(f"⚡ Saved by only processing active regions per chunk!")
    
    return latent"""
    def __init__(self, 
                 crossfade_frames=32,    # Frames for smooth transitions between regions
                 use_crossfade=True):    # Whether to use crossfading or hard boundaries
        self.crossfade_frames = crossfade_frames
        self.use_crossfade = use_crossfade

def create_temporal_mask(start_time, end_time, total_frames, frame_rate=25.6, config=None, 
                        is_first_region=False, is_last_region=False):
    """
    Create temporal mask for a region
    
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds  
        total_frames: Total number of frames in the latent
        frame_rate: Frames per second (AudioLDM default: 25.6)
        config: RegionConfig object
        is_first_region: Whether this is the first region (no fade-in at start)
        is_last_region: Whether this is the last region (no fade-out at end)
    
    Returns:
        torch.Tensor: Binary mask [1, 1, time_frames, 1]
    """
    if config is None:
        config = RegionConfig()
    
    # Convert times to frame indices
    start_frame = int(start_time * frame_rate)
    end_frame = int(end_time * frame_rate)
    
    # Clamp to valid range
    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)
    
    # Create base mask
    mask = torch.zeros(1, 1, total_frames, 1)
    
    if config.use_crossfade and config.crossfade_frames > 0:
        fade_frames = min(config.crossfade_frames, (end_frame - start_frame) // 4)
        
        # For first region: start at 1.0 immediately (no fade-in)
        if is_first_region:
            actual_start = 0  # Start at very beginning
            fade_in_end = actual_start  # No fade-in
        else:
            # Normal fade-in for middle regions
            actual_start = max(0, start_frame - fade_frames // 2)
            fade_in_end = min(actual_start + fade_frames, end_frame)
            if fade_in_end > actual_start:
                fade_in_values = torch.linspace(0, 1, fade_in_end - actual_start)
                mask[0, 0, actual_start:fade_in_end, 0] = fade_in_values
        
        # For last region: end at 1.0 (no fade-out)
        if is_last_region:
            actual_end = total_frames  # End at very end
            fade_out_start = actual_end  # No fade-out
        else:
            # Normal fade-out for middle regions
            actual_end = min(total_frames, end_frame + fade_frames // 2)
            fade_out_start = max(actual_end - fade_frames, fade_in_end)
        
        # Full strength region
        full_start = fade_in_end
        full_end = fade_out_start
        if full_end > full_start:
            mask[0, 0, full_start:full_end, 0] = 1.0
        
        # Fade out (only for non-last regions)
        if not is_last_region and fade_out_start < actual_end:
            fade_out_values = torch.linspace(1, 0, actual_end - fade_out_start)
            mask[0, 0, fade_out_start:actual_end, 0] = fade_out_values
        elif is_last_region:
            # Last region stays at 1.0 until the very end
            mask[0, 0, fade_out_start:actual_end, 0] = 1.0
        
        # First region starts at 1.0 from the very beginning
        if is_first_region:
            mask[0, 0, 0:fade_in_end, 0] = 1.0
            
    else:
        # Hard boundaries - no crossfading
        mask[0, 0, start_frame:end_frame, 0] = 1.0
    
    return mask

def visualize_masks(masks, segment_info, total_duration):
    """Visualize temporal masks"""
    total_frames = masks[0].shape[2]
    time_axis = np.linspace(0, total_duration, total_frames)
    
    plt.figure(figsize=(15, 6))
    
    # Plot individual masks
    plt.subplot(2, 1, 1)
    for i, (mask, (start_time, end_time, prompt)) in enumerate(zip(masks, segment_info)):
        mask_values = mask[0, 0, :, 0].cpu().numpy()
        plt.plot(time_axis, mask_values, label=f'Region {i+1}: "{prompt}"', linewidth=2)
    
    plt.title("Individual Region Masks")
    plt.xlabel("Time (s)")
    plt.ylabel("Mask Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot combined mask
    plt.subplot(2, 1, 2)
    mask_sum = sum(masks)
    combined_values = mask_sum[0, 0, :, 0].cpu().numpy()
    plt.plot(time_axis, combined_values, 'k-', linewidth=2, label='Combined Mask Sum')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Target Value')
    
    plt.title("Combined Mask Sum (should be ~1.0 everywhere)")
    plt.xlabel("Time (s)")
    plt.ylabel("Total Mask Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def get_audio_views(total_frames, window_size=256, stride=64):
    """Get overlapping temporal views for audio (adapted from image MultiDiffusion)"""
    num_windows = (total_frames - window_size) // stride + 1
    views = []
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        if end > total_frames:
            end = total_frames
            start = max(0, end - window_size)
        views.append((start, end))
    return views

def region_multidiffusion_sample(sampler, shape, segment_embeddings, segment_info, 
                                unconditional_conditioning, unconditional_guidance_scale=3.0, 
                                eta=0.1, x_T=None, S=200, config=None, chunk_frames=256, overlap_ratio=0.25):
    """
    Region-based MultiDiffusion sampling (chunked approach like image MultiDiffusion)
    
    Args:
        sampler: DDIM sampler
        shape: Latent shape [batch, channels, time_frames, freq_bins]
        segment_embeddings: List of conditioning embeddings for each region
        segment_info: List of (start_time, end_time, prompt) tuples
        unconditional_conditioning: Unconditional embedding for CFG
        unconditional_guidance_scale: CFG scale
        eta: DDIM eta parameter
        x_T: Initial noise tensor
        S: Number of diffusion steps
        config: RegionConfig object
    
    Returns:
        Denoised latent tensor
    """
    if config is None:
        config = RegionConfig()
    
    model = sampler.model
    device = model.device
    
    # Prepare DDIM schedule
    sampler.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
    
    # Unpack shape
    batch_size, channels, total_frames, freq_bins = shape
    
    # Create temporal masks for each region (no background mask)
    total_duration = total_frames / 25.6  # Convert frames to seconds
    region_masks = []
    
    print(f"🎭 CHUNKED MULTIDIFFUSION: Creating masks for {len(segment_info)} regions (no background)")
    
    # Create extended segment info with proper boundaries
    extended_segment_info = []
    for i, (start_time, end_time, prompt) in enumerate(segment_info):
        # Extend first region to start from 0, last region to end at total_duration
        if i == 0:
            start_time = 0.0
        if i == len(segment_info) - 1:
            end_time = total_duration
            
        extended_segment_info.append((start_time, end_time, prompt))
        
        # Create mask with proper boundary handling
        is_first = (i == 0)
        is_last = (i == len(segment_info) - 1)
        mask = create_temporal_mask(start_time, end_time, total_frames, config=config,
                                  is_first_region=is_first, is_last_region=is_last)
        region_masks.append(mask.to(device))
        print(f"  Region {i+1}: '{prompt}' [{start_time:.1f}s-{end_time:.1f}s] (first={is_first}, last={is_last})")
    
    # NO background mask - just use the region masks directly
    all_masks = region_masks
    all_embeddings = segment_embeddings
    
    print(f"📊 Using {len(all_masks)} region masks (no background)")
    
    # Visualize region masks
    print("📊 Visualizing region masks...")
    visualize_masks([m.cpu() for m in all_masks], extended_segment_info, total_duration)
    
    # Initialize noise
    if x_T is None:
        x_T = torch.randn(shape).to(device)
    
    # Timesteps and reverse order for DDIM
    timesteps = sampler.ddim_timesteps[:S]
    time_sequence = np.flip(timesteps)
    
    # Initialize current latent
    latent = x_T.clone()
    
    print(f"🔄 Starting CORRECT MultiDiffusion with {len(time_sequence)} steps")
    print(f"📊 Will process chunks and only generate for active regions per chunk")
    
    # Calculate overlap frames from ratio
    overlap_frames = int(chunk_frames * overlap_ratio)
    stride = chunk_frames - overlap_frames
    print(f"🪟 Chunk settings: {chunk_frames} frames, {overlap_frames} overlap, stride {stride}")
    
    # Process overlapping chunks
    views = get_audio_views(total_frames, window_size=chunk_frames, stride=stride)
    
    # Main denoising loop (CORRECT MultiDiffusion: chunk-based with region conditioning)
    with tqdm(total=len(time_sequence), desc="Correct MultiDiffusion") as pbar:
        for i, step in enumerate(time_sequence):
            index = len(timesteps) - i - 1
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            
            # Initialize noise prediction accumulator for this timestep
            noise_pred_full = torch.zeros_like(latent)
            count_full = torch.zeros_like(latent)
            
            # Process overlapping chunks
            views = get_audio_views(total_frames, window_size=chunk_frames, stride=stride)
            
            for t_start, t_end in views:
                # Extract chunk from current latent
                latent_chunk = latent[:, :, t_start:t_end, :]  # [1, C, chunk_frames, F]
                
                # Check which regions are active for this chunk
                active_regions = []
                active_embeddings = []
                active_masks_chunk = []
                
                for region_idx, mask in enumerate(all_masks):
                    mask_chunk = mask[:, :, t_start:t_end, :]
                    if mask_chunk.max().item() > 1e-6:  # Has contribution
                        active_regions.append(region_idx)
                        active_embeddings.append(all_embeddings[region_idx])
                        active_masks_chunk.append(mask_chunk)
                
                if len(active_regions) == 0:
                    continue  # Skip if no active regions
                
                # Get noise predictions for active regions on this chunk
                chunk_noise_pred = torch.zeros_like(latent_chunk)
                chunk_mask_sum = torch.zeros_like(latent_chunk)
                
                for embedding, mask_chunk in zip(active_embeddings, active_masks_chunk):
                    # Apply model to this chunk with this region's conditioning
                    if unconditional_guidance_scale != 1.0:
                        # CFG
                        latent_input = torch.cat([latent_chunk, latent_chunk])
                        t_input = torch.cat([t, t])
                        cond_input = torch.cat([unconditional_conditioning, embedding], dim=0)
                        
                        with torch.no_grad():
                            noise_pred_combined = model.apply_model(latent_input, t_input, cond_input)
                        
                        noise_pred_uncond, noise_pred_cond = noise_pred_combined.chunk(2)
                        noise_pred = noise_pred_uncond + unconditional_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        with torch.no_grad():
                            noise_pred = model.apply_model(latent_chunk, t, embedding)
                    
                    # Weight by mask and accumulate for this chunk
                    chunk_noise_pred += noise_pred * mask_chunk
                    chunk_mask_sum += mask_chunk
                
                # Normalize chunk noise prediction
                chunk_noise_pred = chunk_noise_pred / (chunk_mask_sum + 1e-8)
                
                # Accumulate into full noise prediction (overlap averaging)
                noise_pred_full[:, :, t_start:t_end, :] += chunk_noise_pred
                count_full[:, :, t_start:t_end, :] += 1.0
            
            # Average overlapping chunks
            final_noise_pred = noise_pred_full / (count_full + 1e-8)
            
            # Apply DDIM step ONCE to the full latent
            a_t = sampler.ddim_alphas[index]
            a_prev = sampler.ddim_alphas_prev[index]
            sigma_t = sampler.ddim_sigmas[index]
            sqrt_one_minus_at = sampler.ddim_sqrt_one_minus_alphas[index]
            
            # Convert to tensor format (not numpy!)
            a_t = torch.tensor(a_t, device=device)
            a_prev = torch.tensor(a_prev, device=device)
            sigma_t = torch.tensor(sigma_t, device=device)
            sqrt_one_minus_at = torch.tensor(sqrt_one_minus_at, device=device)
            
            # DDIM step on full latent
            pred_x0 = (latent - sqrt_one_minus_at * final_noise_pred) / a_t.sqrt()
            dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * final_noise_pred
            noise = sigma_t * torch.randn_like(latent) if index > 0 else 0.0
            
            latent = a_prev.sqrt() * pred_x0 + dir_xt + noise
            
            pbar.update(1)
    
    return latent

def region_multidiffusion_text_to_audio(model, segment_info, duration, ddim_steps=200,
                                       unconditional_guidance_scale=3.0, ddim_eta=0.1, 
                                       config=None, use_sliding_vae=True, return_intermediates=False,
                                       chunk_frames=256, overlap_ratio=0.25):
    """
    Generate audio using region-based MultiDiffusion
    
    Args:
        model: AudioLDM model
        segment_info: List of (start_time, end_time, prompt) tuples
        duration: Total duration in seconds
        ddim_steps: Number of diffusion steps
        unconditional_guidance_scale: CFG scale
        ddim_eta: DDIM eta parameter
        config: RegionConfig object
        use_sliding_vae: Whether to use sliding window VAE decoding
        return_intermediates: Whether to return intermediate latent and mel results
    
    Returns:
        numpy.ndarray: Generated waveform
        OR if return_intermediates=True:
        tuple: (waveform, latent_samples, mel_spectrogram)
    """
    if config is None:
        config = RegionConfig()
    
    device = model.device
    
    print(f"🎭 REGION MULTIDIFFUSION: Generating {duration}s audio with {len(segment_info)} regions")
    
    # Prepare conditioning
    model.cond_stage_model.embed_mode = "text"
    
    # Get embeddings for each segment
    segment_embeddings = []
    prompts = []
    for start_time, end_time, prompt in segment_info:
        embedding = model.get_learned_conditioning([prompt])
        segment_embeddings.append(embedding)
        prompts.append(prompt)
        print(f"  📝 '{prompt}' [{start_time:.1f}s-{end_time:.1f}s]")
    
    # Unconditional embedding
    unconditional_conditioning = model.get_learned_conditioning([""])
    
    # Setup sampler and shape
    sampler = DDIMSampler(model)
    latent_size = duration_to_latent_t_size(duration)
    shape = [1, model.channels, latent_size, model.latent_f_size]
    
    print(f"📐 Latent shape: {shape}")
    
    # Generate latent using region MultiDiffusion
    start_time = time.time()
    with torch.no_grad():
        samples = region_multidiffusion_sample(
            sampler=sampler,
            shape=shape,
            segment_embeddings=segment_embeddings,
            segment_info=segment_info,
            unconditional_conditioning=unconditional_conditioning,
            unconditional_guidance_scale=unconditional_guidance_scale,
            eta=ddim_eta,
            S=ddim_steps,
            config=config,
            chunk_frames=chunk_frames,
            overlap_ratio=overlap_ratio
        )
    diffusion_time = time.time() - start_time
    print(f"⏱️  Diffusion time: {diffusion_time:.1f}s")
    
    # VAE decode
    print("🪟 VAE decoding...")
    start_time = time.time()
    
    if use_sliding_vae and samples.shape[2] > 512:
        # Use sliding window VAE decode for long sequences
        from utils.vaeutils import sliding_window_vae_decode
        mel = sliding_window_vae_decode(model, samples, window_size=256, overlap=64)
    else:
        # Standard VAE decode
        with torch.no_grad():
            mel = model.decode_first_stage(samples)
    
    vae_time = time.time() - start_time
    print(f"⏱️  VAE time: {vae_time:.1f}s")
    
    # Vocoder
    print("🔊 Vocoder...")
    start_time = time.time()
    with torch.no_grad():
        waveform = model.mel_spectrogram_to_waveform(mel)
    vocoder_time = time.time() - start_time
    print(f"⏱️  Vocoder time: {vocoder_time:.1f}s")
    
    total_time = diffusion_time + vae_time + vocoder_time
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    # Convert to numpy
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy().squeeze()
    
    if return_intermediates:
        return waveform, samples, mel
    else:
        return waveform

def create_test_segments(duration, num_segments=3):
    """Create test segments for demonstration"""
    segment_duration = duration / num_segments
    
    test_prompts = [
        "90s rock song with electric guitar and heavy drums",
        "smooth jazz piano solo with soft drums", 
        "electronic dance music with synthesizers and bass"
    ]
    
    segments = []
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        prompt = test_prompts[i % len(test_prompts)]
        segments.append((start_time, end_time, prompt))
    
    return segments

def analyze_region_coherence(waveform, segment_info, sr=16000):
    """Analyze coherence between regions"""
    print("🔍 Analyzing region coherence...")
    
    # Calculate transition points
    transition_times = []
    for i in range(len(segment_info) - 1):
        transition_time = segment_info[i][1]  # End time of current segment
        transition_times.append(transition_time)
    
    # Analyze amplitude continuity at transitions
    for i, transition_time in enumerate(transition_times):
        transition_sample = int(transition_time * sr)
        
        # Get samples around transition
        window = sr // 10  # 100ms window
        start_idx = max(0, transition_sample - window)
        end_idx = min(len(waveform), transition_sample + window)
        
        if end_idx > start_idx:
            transition_audio = waveform[start_idx:end_idx]
            
            # Calculate RMS in before/after regions
            mid_point = len(transition_audio) // 2
            before_rms = np.sqrt(np.mean(transition_audio[:mid_point]**2))
            after_rms = np.sqrt(np.mean(transition_audio[mid_point:]**2))
            
            rms_ratio = after_rms / (before_rms + 1e-8)
            
            print(f"  Transition {i+1} at {transition_time:.1f}s:")
            print(f"    Before RMS: {before_rms:.4f}")
            print(f"    After RMS: {after_rms:.4f}")
            print(f"    Ratio: {rms_ratio:.2f}")
