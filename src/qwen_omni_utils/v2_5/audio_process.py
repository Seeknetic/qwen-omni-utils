import base64
from io import BytesIO

import audioread
import av
import librosa
import numpy as np


SAMPLE_RATE=16000
def _check_if_video_has_audio(video): # video is a file path or BytesIO object
    """Check if video has audio streams. Supports both file paths and BytesIO objects."""

    container = av.open(video)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    container.close()
    if not audio_streams:
        return False
    return True


def process_audio_info(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool | list[bool]):
    """
    Read and process audio info

    Support dict keys:

    type = audio
    - audio
    - audio_start
    - audio_end

    type = video
    - video
    - video_start
    - video_end
    
    Args:
        conversations: List of conversations, each conversation is a list of messages
        use_audio_in_video: Boolean or list of booleans. If bool, applies to all conversations.
                           If list, len(use_audio_in_video) == len(conversations)
    """
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    
    # Handle use_audio_in_video parameter
    if isinstance(use_audio_in_video, bool):
        # If bool, create a list with the same value for each conversation
        use_audio_flags = [use_audio_in_video] * len(conversations)
    elif isinstance(use_audio_in_video, list):
        # If list, validate length matches conversations
        if len(use_audio_in_video) != len(conversations):
            raise ValueError(f"Length of use_audio_in_video ({len(use_audio_in_video)}) must match length of conversations ({len(conversations)})")
        use_audio_flags = use_audio_in_video
    else:
        raise TypeError("use_audio_in_video must be bool or list[bool]")
    
    for conv_idx, conversation in enumerate(conversations):
        current_use_audio = use_audio_flags[conv_idx]
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele or "audio_url" in ele:
                        path = ele.get("audio", ele.get("audio_url"))
                        audio_start = ele.get("audio_start", 0.0)
                        audio_end = ele.get("audio_end", None)
                        if isinstance(path, np.ndarray):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")
                            audios.append(
                                path[int(SAMPLE_RATE * audio_start) : None if audio_end is None else int(SAMPLE_RATE * audio_end)]
                            )
                            continue
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = BytesIO(base64.b64decode(base64_data))
                        elif path.startswith("http://") or path.startswith("https://"):
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                elif current_use_audio and ele["type"] == "video":
                    if "video" in ele or "video_url" in ele:
                        path = ele.get("video", ele.get("video_url"))
                        if isinstance(path, bytes):
                            path = BytesIO(path)
                        audio_start = ele.get("video_start", 0.0)
                        audio_end = ele.get("video_end", None)
                        if not _check_if_video_has_audio(path):
                            continue
                        if isinstance(path, str) and (path.startswith("http://") or path.startswith("https://")):
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif isinstance(path, str) and path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                else:
                    continue
                
                # Handle BytesIO objects containing video data (e.g., MP4)
                if isinstance(data, BytesIO):
                    # Use av library to extract audio from video BytesIO
                    data.seek(0)  # Reset position
                    container = av.open(data)
                    
                    # Find audio stream
                    audio_stream = None
                    for stream in container.streams:
                        if stream.type == "audio":
                            audio_stream = stream
                            break
                    
                    if audio_stream is None:
                        container.close()
                        continue
                    
                    # Extract audio data
                    audio_data = []
                    start_pts = int(audio_start * audio_stream.time_base.denominator / audio_stream.time_base.numerator) if audio_start > 0 else None
                    end_pts = int(audio_end * audio_stream.time_base.denominator / audio_stream.time_base.numerator) if audio_end is not None else None
                    
                    for frame in container.decode(audio_stream):
                        if start_pts is not None and frame.pts < start_pts:
                            continue
                        if end_pts is not None and frame.pts > end_pts:
                            break
                        
                        # Convert frame to numpy array and resample to target sample rate
                        frame_array = frame.to_ndarray()
                        if frame_array.ndim > 1:
                            # Convert to mono by averaging channels
                            frame_array = np.mean(frame_array, axis=0)
                        audio_data.append(frame_array)
                    
                    container.close()
                    
                    if audio_data:
                        # Concatenate all audio frames
                        audio = np.concatenate(audio_data)
                        
                        # Resample if necessary
                        if audio_stream.sample_rate != SAMPLE_RATE:
                            audio = librosa.resample(audio, orig_sr=audio_stream.sample_rate, target_sr=SAMPLE_RATE)
                        
                        audios.append(audio)
                else:
                    # Use librosa for file paths and other supported formats
                    audio = librosa.load(
                        data,
                        sr=SAMPLE_RATE,
                        offset=audio_start,
                        duration=(audio_end - audio_start) if audio_end is not None else None,
                    )[0]
                    audios.append(audio)
    if len(audios) == 0:
        audios = None
    return audios
