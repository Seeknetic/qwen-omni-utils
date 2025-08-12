# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `qwen-omni-utils`, a Python utility package for processing multimodal data (images, videos, audio) for integration with Qwen-Omni language models. The package provides helper functions to process visual and audio information from various sources (local files, URLs, base64, PIL Images, numpy arrays) and prepare them for model consumption.

## Development Commands

### Building and Installing
```bash
pip install -e .                    # Install package in development mode
pip install qwen-omni-utils          # Install from PyPI
```

### Code Quality
```bash
ruff check src/                     # Run linting with ruff
ruff format src/                    # Format code with ruff
```

### Package Building
```bash
python -m build                     # Build wheel and source distribution
```

## Architecture Overview

The package follows a simple, modular architecture centered around processing multimodal data:

### Core Structure
- **Main entry point**: `src/qwen_omni_utils/__init__.py` - imports everything from v2_5
- **Processing modules**: Located in `src/qwen_omni_utils/v2_5/`
  - `audio_process.py` - Audio processing utilities
  - `vision_process.py` - Image and video processing utilities
  - `__init__.py` - Main processing functions and exports

### Key Functions
- `process_mm_info()` - Main function that processes multimodal information (audio + vision)
- `process_vision_info()` - Processes images and videos from various sources
- `process_audio_info()` - Processes audio from files or numpy arrays

### Data Flow
1. Input conversations contain multimodal elements (image, video, audio)
2. Elements are extracted and validated
3. Media is fetched from various sources (file://, http://, data:, PIL objects)
4. Content is processed and resized according to model constraints
5. Processed data is returned in formats ready for model consumption

### Vision Processing Features
- Smart resizing with configurable pixel constraints (MIN_PIXELS, MAX_PIXELS)
- Multiple video backends: torchcodec (preferred), decord, torchvision (fallback)
- Frame sampling with configurable FPS
- Support for various input formats: local files, URLs, base64, PIL Images
- Automatic format conversion (RGBA → RGB with white background)

### Audio Processing Features
- 16kHz sample rate processing using librosa
- Support for numpy arrays, file paths, and URLs
- Audio extraction from videos when `use_audio_in_video=True`
- Time-based clipping with start/end parameters

### Configuration
- Environment variable `VIDEO_MAX_PIXELS` controls maximum video token inputs (default: 128K * 28 * 28 * 0.9)
- Environment variable `FORCE_QWENVL_VIDEO_READER` forces specific video backend
- Environment variable `TORCHCODEC_NUM_THREADS` controls torchcodec threading (default: 8)

### Dependencies
- Core: `requests`, `pillow`, `av`, `packaging`, `librosa`
- Optional: `decord` for better video processing
- Dev dependencies: `torch`, `torchvision`, `torchaudio`

### Code Style
- Line length: 119 characters
- Uses ruff for linting and formatting
- Double quotes preferred
- Space-based indentation