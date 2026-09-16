"""ComfyUI LTX 2.3 workflow generator with IAMCCS audio extension.

Two workflow types:
1. Initial segment: generates first video+audio clip from reference image
2. Extension segment: extends previous latents with new frames conditioned
   on audio, using IAMCCS latent overlap for seamless continuity

Uses the dev model with native joint audio-video generation.
"""

import json
from pathlib import Path
from . import config


def build_initial_segment_workflow(
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted, watermarks",
    image_path: str = "",
    audio_path: str = "",
    checkpoint: str = "ltx-2.3-22b-dev.safetensors",
    lora_name: str = "ltx-2.3-22b-distilled-lora-384.safetensors",
    lora_strength: float = 0.4,
    text_encoder: str = "comfy_gemma_3_12B_it/gemma_3_12B_it_fp4_mixed.safetensors",
    width: int = 544,
    height: int = 960,
    num_frames: int = 241,
    fps: float = 24.0,
    video_cfg: float = 3.0,
    audio_cfg: float = 3.0,
    i2v_strength: float = 0.85,
    img_compression: int = 35,
    steps: int = 20,
    seed: int = 42,
    two_stage: bool = True,
    filename_prefix: str = "ugc_seg0",
) -> dict:
    """Build workflow for audio-conditioned video generation.

    The TTS audio is encoded via Audio VAE and drives lip-sync.
    Two-stage option adds 2x latent upscale for 1080p output.
    """

    workflow = {
        # === Model Loading ===
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora_name, "strength_model": lora_strength}},
        "3": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": text_encoder, "ckpt_name": checkpoint, "device": "default"}},
        "4": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": checkpoint}},

        # === Text Encoding ===
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["3", 0]}},

        # === Image + Audio ===
        "20": {"class_type": "LoadImage", "inputs": {"image": image_path}},
        "21": {"class_type": "LTXVPreprocess", "inputs": {"image": ["20", 0], "img_compression": img_compression}},
        "22": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width, "height": height, "length": num_frames, "batch_size": 1}},

        # Audio conditioning from TTS audio
        "23": {"class_type": "LoadAudio", "inputs": {"audio": audio_path}},
        "24": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["23", 0], "audio_vae": ["4", 0]}},

        # === Conditioning ===
        "30": {"class_type": "LTXVImgToVideoConditionOnly", "inputs": {"vae": ["1", 2], "image": ["21", 0], "latent": ["22", 0], "strength": i2v_strength}},
        "31": {"class_type": "LTXVConditioning", "inputs": {"positive": ["10", 0], "negative": ["11", 0], "frame_rate": fps}},
        # Concat I2V video latent + encoded audio latent
        "32": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["30", 0], "audio_latent": ["24", 0]}},

        # === Sampling ===
        "40": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "41": {"class_type": "LTXVScheduler", "inputs": {"steps": steps, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1}},
        "42": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "43": {"class_type": "CFGGuider", "inputs": {"model": ["2", 0], "positive": ["31", 0], "negative": ["31", 1], "cfg": video_cfg}},
        "44": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["40", 0], "guider": ["43", 0], "sampler": ["42", 0],
            "sigmas": ["41", 0], "latent_image": ["32", 0],
        }},

    }

    s2_sigmas = "0.85, 0.7250, 0.5, 0.3, 0.15, 0.0"

    if two_stage:
        # === Stage 2: Latent Upscale + Refine ===
        workflow["50"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["44", 0]}}
        workflow["51"] = {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"}}
        workflow["52"] = {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["50", 0], "upscale_model": ["51", 0], "vae": ["1", 2]}}
        workflow["53"] = {"class_type": "LTXVImgToVideoConditionOnly", "inputs": {"vae": ["1", 2], "image": ["21", 0], "latent": ["52", 0], "strength": i2v_strength}}
        workflow["54"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["53", 0], "audio_latent": ["50", 1]}}
        workflow["55"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + 1}}
        workflow["56"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": s2_sigmas}}
        workflow["57"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        workflow["58"] = {"class_type": "CFGGuider", "inputs": {"model": ["2", 0], "positive": ["31", 0], "negative": ["31", 1], "cfg": 1.0}}
        workflow["59"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["55", 0], "guider": ["58", 0], "sampler": ["57", 0],
            "sigmas": ["56", 0], "latent_image": ["54", 0],
        }}
        decode_source = "59"
    else:
        decode_source = "44"

    # === Decode ===
    workflow["60"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": [decode_source, 0]}}
    workflow["61"] = {"class_type": "VAEDecodeTiled", "inputs": {
        "samples": ["60", 0], "vae": ["1", 2],
        "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8,
    }}
    workflow["62"] = {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["60", 1], "audio_vae": ["4", 0]}}

    # === Output ===
    workflow["70"] = {"class_type": "CreateVideo", "inputs": {"images": ["61", 0], "audio": ["62", 0], "fps": fps}}
    workflow["71"] = {"class_type": "SaveVideo", "inputs": {"video": ["70", 0], "filename_prefix": filename_prefix, "format": "mp4", "codec": "h264"}}

    return workflow


def build_talking_head_workflow(
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted, watermarks",
    image_path: str = "",
    checkpoint: str = "ltx-2.3-22b-dev.safetensors",
    lora_name: str = "ltx-2.3-22b-distilled-lora-384.safetensors",
    lora_strength: float = 0.4,
    text_encoder: str = "comfy_gemma_3_12B_it/gemma_3_12B_it_fp4_mixed.safetensors",
    width: int = 544,
    height: int = 960,
    num_frames: int = 241,
    fps: float = 24.0,
    video_cfg: float = 3.0,
    audio_cfg: float = 3.0,
    i2v_strength: float = 0.85,
    img_compression: int = 35,
    steps: int = 20,
    seed: int = 42,
    two_stage: bool = True,
    filename_prefix: str = "ugc_talking_head",
    **kwargs,
) -> dict:
    """Build standard talking head workflow (no external audio — model generates AV jointly).

    This is the fallback for when we don't have pre-generated TTS audio.
    The model generates both video and audio from the text prompt.
    """

    s2_sigmas = "0.85, 0.7250, 0.5, 0.3, 0.15, 0.0"

    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora_name, "strength_model": lora_strength}},
        "3": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": text_encoder, "ckpt_name": checkpoint, "device": "default"}},
        "4": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": checkpoint}},

        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["3", 0]}},

        "20": {"class_type": "LoadImage", "inputs": {"image": image_path}},
        "21": {"class_type": "LTXVPreprocess", "inputs": {"image": ["20", 0], "img_compression": img_compression}},
        "22": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width, "height": height, "length": num_frames, "batch_size": 1}},
        "23": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"audio_vae": ["4", 0], "frames_number": num_frames, "frame_rate": fps, "batch_size": 1}},

        "30": {"class_type": "LTXVImgToVideoConditionOnly", "inputs": {"vae": ["1", 2], "image": ["21", 0], "latent": ["22", 0], "strength": i2v_strength}},
        "31": {"class_type": "LTXVConditioning", "inputs": {"positive": ["10", 0], "negative": ["11", 0], "frame_rate": fps}},
        "32": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["30", 0], "audio_latent": ["23", 0]}},

        "40": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "41": {"class_type": "LTXVScheduler", "inputs": {"steps": steps, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1}},
        "42": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "43": {"class_type": "CFGGuider", "inputs": {"model": ["2", 0], "positive": ["31", 0], "negative": ["31", 1], "cfg": video_cfg}},
        "44": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["40", 0], "guider": ["43", 0], "sampler": ["42", 0],
            "sigmas": ["41", 0], "latent_image": ["32", 0],
        }},
    }

    if two_stage:
        workflow["50"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["44", 0]}}
        workflow["51"] = {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"}}
        workflow["52"] = {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["50", 0], "upscale_model": ["51", 0], "vae": ["1", 2]}}
        workflow["53"] = {"class_type": "LTXVImgToVideoConditionOnly", "inputs": {"vae": ["1", 2], "image": ["21", 0], "latent": ["52", 0], "strength": i2v_strength}}
        workflow["54"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["53", 0], "audio_latent": ["50", 1]}}
        workflow["55"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + 1}}
        workflow["56"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": s2_sigmas}}
        workflow["57"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        workflow["58"] = {"class_type": "CFGGuider", "inputs": {"model": ["2", 0], "positive": ["31", 0], "negative": ["31", 1], "cfg": 1.0}}
        workflow["59"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["55", 0], "guider": ["58", 0], "sampler": ["57", 0],
            "sigmas": ["56", 0], "latent_image": ["54", 0],
        }}
        decode_source = "59"
    else:
        decode_source = "44"

    workflow["60"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": [decode_source, 0]}}
    workflow["61"] = {"class_type": "VAEDecodeTiled", "inputs": {
        "samples": ["60", 0], "vae": ["1", 2],
        "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8,
    }}
    workflow["62"] = {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["60", 1], "audio_vae": ["4", 0]}}
    workflow["70"] = {"class_type": "CreateVideo", "inputs": {"images": ["61", 0], "audio": ["62", 0], "fps": fps}}
    workflow["71"] = {"class_type": "SaveVideo", "inputs": {"video": ["70", 0], "filename_prefix": filename_prefix, "format": "mp4", "codec": "h264"}}

    return workflow


def save_workflow(workflow: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(workflow, f, indent=2)
    return output_path
