# # whisper_service.py
# import os
# import logging
# from faster_whisper import WhisperModel

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def check_gpu_available():
#     """Check if CUDA/GPU is available without importing torch"""
#     try:
#         import subprocess
#         result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
#         return result.returncode == 0
#     except:
#         return False

# def get_optimal_model_settings():
#     """
#     Choose the best model and settings based on available hardware
#     Returns: (model_size, device, compute_type)
#     """
#     has_gpu = check_gpu_available()
    
#     if has_gpu:
#         logger.info("GPU detected - using CUDA")
#         return "small", "cuda", "float16"  # Good balance of speed/accuracy
#     else:
#         logger.info("Using CPU - optimizing for efficiency")
#         return "small", "cpu", "int8"  # Better than base model, still fast

# # Initialize model with optimal settings
# try:
#     model_size, device, compute_type = get_optimal_model_settings()
    
#     # Additional CPU optimizations to prevent overheating
#     if device == "cpu":
#         model = WhisperModel(
#             model_size, 
#             device=device, 
#             compute_type=compute_type,
#             cpu_threads=2,  # Limit CPU threads to prevent overheating
#             num_workers=1   # Single worker to reduce load
#         )
#     else:
#         model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
#     logger.info(f"Whisper model loaded: {model_size} on {device}")
    
# except Exception as e:
#     logger.error(f"Failed to load optimal model: {e}")
#     # Ultra-light fallback for overheating prevention
#     model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=1)
#     model_size = "tiny"
#     logger.info("Using fallback tiny model")

# def transcribe_audio(file_path: str, language=None) -> str:
#     """
#     Optimized transcribe function with better accuracy and speed
    
#     Args:
#         file_path: Path to audio file
#         language: Language code (e.g., 'en' for English) - improves accuracy significantly
    
#     Returns:
#         Transcribed text
#     """
#     try:
#         logger.info(f"Starting transcription with {model_size} model")
        
#         # Optimized parameters for better accuracy and speed
#         segments, info = model.transcribe(
#             file_path,
#             language=language or "en",  # Default to English if not specified
#             beam_size=3,        # Good balance of speed and accuracy
#             best_of=3,          # Multiple candidates for better results
#             temperature=0.0,    # Deterministic output
#             compression_ratio_threshold=2.4,  # Skip repetitive segments
#             log_prob_threshold=-1.0,          # Skip low-confidence segments
#             no_speech_threshold=0.6,          # Better silence detection
#             condition_on_previous_text=True,  # Use context for better accuracy
#             initial_prompt="This is a meeting or conversation transcript with clear speech.",
#             word_timestamps=False,  # Disabled for speed
#             vad_filter=True,       # Voice Activity Detection - removes silence
#             vad_parameters=dict(min_silence_duration_ms=500)  # Remove long silences
#         )
        
#         # Collect and clean transcript
#         transcript_parts = []
#         for segment in segments:
#             text = segment.text.strip()
#             if text and len(text) > 1:  # Skip very short segments
#                 transcript_parts.append(text)
        
#         transcript = " ".join(transcript_parts)
        
#         logger.info(f"Transcription complete: {len(transcript)} chars, detected language: {info.language}")
#         return transcript
        
#     except Exception as e:
#         error_msg = f"Transcription failed: {str(e)}"
#         logger.error(error_msg)
#         return f"Error: {error_msg}"

# def get_model_info():
#     """Get info about current model"""
#     return {
#         "model_size": model_size,
#         "device": "GPU" if check_gpu_available() else "CPU",
#         "gpu_available": check_gpu_available()
#     }

# whisper_service.py
# import os
# import logging
# import psutil
# import time
# from faster_whisper import WhisperModel

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def check_gpu_available():
#     """Check if CUDA/GPU is available without importing torch"""
#     try:
#         import subprocess
#         result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
#         return result.returncode == 0
#     except:
#         return False

# def get_system_friendly_settings():
#     """
#     Ultra-conservative settings for 8th gen i7 + 8GB RAM to prevent overheating
#     """
#     # Force CPU with minimal resource usage
#     logger.info("Using laptop-friendly settings to prevent overheating")
#     return "tiny", "cpu", "int8"  # Smallest, coolest model

# def monitor_cpu_temp():
#     """Basic CPU usage monitoring"""
#     try:
#         cpu_percent = psutil.cpu_percent(interval=1)
#         if cpu_percent > 80:
#             logger.warning(f"High CPU usage detected: {cpu_percent}%")
#         return cpu_percent
#     except:
#         return 0

# # Initialize with ULTRA-LIGHT settings for your laptop
# try:
#     model_size, device, compute_type = get_system_friendly_settings()
    
#     # Extremely conservative CPU settings to prevent overheating
#     model = WhisperModel(
#         model_size,                    # tiny model - fastest, coolest
#         device=device,                 # CPU only
#         compute_type=compute_type,     # int8 - most efficient
#         cpu_threads=2,                 # Limit to 2 threads max
#         num_workers=1,                 # Single worker
#         download_root=None,            # Default cache
#         local_files_only=False         # Allow download if needed
#     )
    
#     logger.info(f"Laptop-friendly Whisper loaded: {model_size} model with 2 CPU threads")
    
# except Exception as e:
#     logger.error(f"Failed to load model: {e}")
#     # Emergency fallback - absolute minimum
#     model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=1)
#     model_size = "tiny"
#     logger.info("Using emergency minimal settings")

# def transcribe_audio_cool(file_path: str, language=None, max_duration=None) -> str:
#     """
#     Cool-running transcription optimized for laptops
#     Includes breaks to prevent overheating
    
#     Args:
#         file_path: Path to audio file
#         language: Language code (e.g., 'en' for English)
#         max_duration: Max duration to process (None = all)
    
#     Returns:
#         Transcribed text
#     """
#     try:
#         logger.info(f"Starting COOL transcription - laptop safe mode")
        
#         # Check initial CPU usage
#         initial_cpu = monitor_cpu_temp()
#         logger.info(f"Starting CPU usage: {initial_cpu}%")
        
#         # Ultra-light transcription parameters
#         segments, info = model.transcribe(
#             file_path,
#             language=language or "en",
#             beam_size=1,                    # Minimum computation
#             best_of=1,                      # Single pass
#             temperature=0.0,                # Deterministic
#             compression_ratio_threshold=4.0, # Very lenient
#             log_prob_threshold=-2.0,        # Very lenient  
#             no_speech_threshold=0.3,        # Lower threshold
#             condition_on_previous_text=False, # Disable for speed
#             initial_prompt=None,            # No prompt processing
#             word_timestamps=False,          # Disabled
#             vad_filter=True,               # Keep for efficiency
#             vad_parameters=dict(
#                 min_silence_duration_ms=2000,  # Remove long silences
#                 threshold=0.4
#             )
#         )
        
#         # Process segments with cooling breaks
#         transcript_parts = []
#         segment_count = 0
        
#         for segment in segments:
#             text = segment.text.strip()
#             if text:
#                 transcript_parts.append(text)
#                 segment_count += 1
                
#                 # Add cooling break every 50 segments
#                 if segment_count % 50 == 0:
#                     logger.info(f"Processed {segment_count} segments - cooling break...")
#                     time.sleep(0.5)  # Half second break
                    
#                     # Check CPU usage
#                     current_cpu = psutil.cpu_percent()
#                     if current_cpu > 85:
#                         logger.warning(f"High CPU usage {current_cpu}% - longer cooling break")
#                         time.sleep(2)  # Longer break if overheating
        
#         transcript = " ".join(transcript_parts)
        
#         # Final CPU check
#         final_cpu = psutil.cpu_percent()
#         logger.info(f"Transcription complete: {len(transcript)} chars")
#         logger.info(f"Final CPU usage: {final_cpu}% (started at {initial_cpu}%)")
        
#         return transcript
        
#     except Exception as e:
#         error_msg = f"Cool transcription failed: {str(e)}"
#         logger.error(error_msg)
#         return f"Error: {error_msg}"

# def transcribe_audio_chunks_cool(file_path: str, language=None, chunk_duration=120) -> str:
#     """
#     Process long audio in small chunks with cooling breaks
#     Perfect for 30+ minute files on laptops
    
#     Args:
#         file_path: Path to audio file
#         language: Language code
#         chunk_duration: Duration of each chunk in seconds (default: 2 minutes)
#     """
#     try:
#         import subprocess
#         import tempfile
        
#         logger.info(f"Processing long audio in {chunk_duration}s chunks with cooling")
        
#         # Get audio duration first
#         result = subprocess.run([
#             'ffprobe', '-v', 'quiet', '-show_entries', 
#             'format=duration', '-of', 'csv=p=0', file_path
#         ], capture_output=True, text=True)
        
#         total_duration = float(result.stdout.strip())
#         logger.info(f"Total audio duration: {total_duration:.1f} seconds")
        
#         # Process in chunks
#         all_transcripts = []
#         current_time = 0
#         chunk_num = 0
        
#         while current_time < total_duration:
#             chunk_num += 1
#             end_time = min(current_time + chunk_duration, total_duration)
            
#             logger.info(f"Processing chunk {chunk_num}: {current_time:.1f}s - {end_time:.1f}s")
            
#             # Extract chunk to temporary file
#             with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
#                 subprocess.run([
#                     'ffmpeg', '-y', '-i', file_path,
#                     '-ss', str(current_time), '-t', str(chunk_duration),
#                     '-ar', '16000', '-ac', '1',  # Downsample for speed
#                     tmp_file.name
#                 ], capture_output=True)
                
#                 # Transcribe chunk
#                 chunk_transcript = transcribe_audio_cool(tmp_file.name, language)
#                 all_transcripts.append(chunk_transcript)
                
#                 # Clean up
#                 os.unlink(tmp_file.name)
            
#             current_time += chunk_duration
            
#             # Cooling break between chunks
#             logger.info(f"Chunk {chunk_num} complete - cooling break...")
#             time.sleep(3)  # 3-second break between chunks
            
#             # Monitor CPU
#             cpu_usage = psutil.cpu_percent()
#             if cpu_usage > 80:
#                 logger.warning(f"High CPU {cpu_usage}% - extended cooling break")
#                 time.sleep(10)  # Longer break if overheating
        
#         final_transcript = " ".join(all_transcripts)
#         logger.info(f"All chunks processed! Total transcript: {len(final_transcript)} chars")
#         return final_transcript
        
#     except Exception as e:
#         error_msg = f"Chunked transcription failed: {str(e)}"
#         logger.error(error_msg)
#         return f"Error: {error_msg}"

# # Main function - use the cool version
# def transcribe_audio(file_path: str, language=None) -> str:
#     """
#     Main transcription function - automatically uses laptop-safe method
#     """
#     try:
#         # Check file size/duration to decide method
#         file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
#         if file_size_mb > 50:  # Large file - use chunking
#             logger.info(f"Large file detected ({file_size_mb:.1f}MB) - using chunked processing")
#             return transcribe_audio_chunks_cool(file_path, language)
#         else:
#             logger.info(f"Standard file ({file_size_mb:.1f}MB) - using cool transcription")
#             return transcribe_audio_cool(file_path, language)
            
#     except Exception as e:
#         logger.error(f"Transcription method selection failed: {e}")
#         # Fallback to simplest method
#         return transcribe_audio_cool(file_path, language)

# def get_model_info():
#     """Get info about current model"""
#     return {
#         "model_size": model_size,
#         "device": "CPU (laptop-safe)",
#         "threads": "2 (cool running)",
#         "gpu_available": False,
#         "optimized_for": "8th gen i7 + 8GB RAM"
#     }

# def check_system_health():
#     """Check if system is running cool"""
#     try:
#         cpu_percent = psutil.cpu_percent(interval=1)
#         memory_percent = psutil.virtual_memory().percent
        
#         status = "Good"
#         if cpu_percent > 80 or memory_percent > 85:
#             status = "Hot - recommend cooling break"
        
#         return {
#             "cpu_usage": f"{cpu_percent:.1f}%",
#             "memory_usage": f"{memory_percent:.1f}%", 
#             "status": status
#         }
#     except:
#         return {"status": "Unable to monitor"}



# whisper_service.py - ULTRA FAST VERSION
# whisper_service.py - EMERGENCY COOL & FAST VERSION

# whisper_service.py - NO FFMPEG REQUIRED VERSION
import os
import logging
import time
from faster_whisper import WhisperModel
import wave
import audioop

# Minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# EMERGENCY COOL MODEL - No preprocessing needed
try:
    model = WhisperModel(
        "tiny.en",        # English-only tiny model
        device="cpu",
        compute_type="int8",
        cpu_threads=1,    # Single thread - cool running
        num_workers=1
    )
    logger.warning("🧊 COOL MODE: tiny.en model loaded (no FFmpeg needed)")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    model = None

def check_audio_file(file_path: str) -> dict:
    """
    Check audio file properties without FFmpeg
    """
    try:
        if file_path.lower().endswith('.wav'):
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / sample_rate
                channels = wav_file.getnchannels()
                
                return {
                    'duration': duration,
                    'sample_rate': sample_rate,
                    'channels': channels,
                    'format': 'WAV'
                }
        else:
            # For non-WAV files, estimate based on file size
            file_size = os.path.getsize(file_path)
            estimated_duration = file_size / 32000  # Rough estimate
            return {
                'duration': estimated_duration,
                'sample_rate': 'unknown',
                'channels': 'unknown', 
                'format': os.path.splitext(file_path)[1].upper()
            }
    except Exception as e:
        logger.error(f"Audio check failed: {e}")
        return {'duration': 0, 'format': 'unknown'}

def transcribe_direct_cool(file_path: str, language="en") -> str:
    """
    Direct transcription without any preprocessing - maximum compatibility
    """
    if not model:
        return "Error: Whisper model not available"
        
    try:
        logger.warning("❄️ DIRECT COOL TRANSCRIPTION - No preprocessing")
        start_time = time.time()
        
        # Check file info
        audio_info = check_audio_file(file_path)
        duration_min = audio_info['duration'] / 60
        logger.warning(f"🎵 Processing {duration_min:.1f} minute {audio_info['format']} file")
        
        # MINIMAL SETTINGS - Let Whisper handle everything
        segments, info = model.transcribe(
            file_path,
            language=language,
            beam_size=1,                    # Minimum computation
            best_of=1,                      # Single candidate
            temperature=0.0,                # Deterministic
            compression_ratio_threshold=10.0, # Lenient
            log_prob_threshold=-3.0,        # Lenient
            no_speech_threshold=0.2,        # Process most audio
            condition_on_previous_text=False, # No context for speed
            initial_prompt=None,            # No prompt
            word_timestamps=False,          # Disabled for speed
            vad_filter=True,               # Built-in silence removal
            vad_parameters=dict(
                min_silence_duration_ms=2000  # Remove 2+ second silences
            )
        )
        
        # Collect transcript with cooling breaks
        transcript_parts = []
        segment_count = 0
        
        for segment in segments:
            text = segment.text.strip()
            if text:
                transcript_parts.append(text)
                segment_count += 1
                
                # Cooling break every 20 segments
                if segment_count % 20 == 0:
                    time.sleep(0.5)  # 500ms cooling break
                    logger.warning(f"❄️ Processed {segment_count} segments - staying cool...")
        
        transcript = " ".join(transcript_parts)
        
        elapsed = time.time() - start_time
        logger.warning(f"⚡ Direct transcription complete: {elapsed:.1f}s")
        logger.warning(f"📝 Generated: {len(transcript)} characters")
        
        return transcript
        
    except Exception as e:
        error_msg = f"Direct transcription failed: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"

def transcribe_with_breaks(file_path: str, language="en", break_interval=30) -> str:
    """
    Transcribe with regular cooling breaks - safest for laptops
    """
    try:
        logger.warning(f"🧊 BREAK MODE: {break_interval}s processing intervals")
        
        # Start transcription in segments with breaks
        if not model:
            return "Error: Model not available"
            
        # Get basic file info
        audio_info = check_audio_file(file_path)
        
        # For long files, warn about processing time
        if audio_info['duration'] > 1500:  # 25+ minutes
            logger.warning("⚠️ Long file detected - this will take several minutes with cooling breaks")
        
        # Direct transcription with extended breaks
        segments, info = model.transcribe(
            file_path,
            language=language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            compression_ratio_threshold=15.0,
            log_prob_threshold=-2.5,
            no_speech_threshold=0.3,
            condition_on_previous_text=False,
            word_timestamps=False,
            vad_filter=True
        )
        
        # Process with extended cooling breaks
        transcript_parts = []
        segment_count = 0
        last_break_time = time.time()
        
        for segment in segments:
            text = segment.text.strip()
            if text:
                transcript_parts.append(text)
                segment_count += 1
                
                # Regular cooling breaks based on time
                current_time = time.time()
                if current_time - last_break_time >= break_interval:
                    logger.warning(f"❄️ Cooling break after {segment_count} segments...")
                    time.sleep(2)  # 2-second cooling break
                    last_break_time = current_time
        
        transcript = " ".join(transcript_parts)
        logger.warning(f"🧊 Break-mode transcription complete: {len(transcript)} chars")
        return transcript
        
    except Exception as e:
        return f"Break-mode transcription failed: {e}"

# MAIN FUNCTIONS - NO FFMPEG REQUIRED
def transcribe_audio(file_path: str, language="en") -> str:
    """
    Main transcription function - works with any audio file Whisper supports
    """
    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"
        
    try:
        # Check file size for strategy
        file_size_mb = os.path.getsize(file_path) / 1024 / 1024
        logger.warning(f"🎵 Processing {file_size_mb:.1f}MB file")
        
        if file_size_mb > 50:  # Very large files
            logger.warning("🧊 Large file - using extended cooling breaks")
            return transcribe_with_breaks(file_path, language, break_interval=20)
        else:
            logger.warning("❄️ Standard file - using direct cool transcription")
            return transcribe_direct_cool(file_path, language)
            
    except Exception as e:
        return f"Transcription failed: {e}"

def transcribe_first_portion(file_path: str, language="en") -> str:
    """
    🎬 DEMO MODE: Transcribe just the beginning for quick demo
    Whisper will naturally process the first portion fastest
    """
    try:
        logger.warning("🎬 DEMO MODE: Processing beginning of file for quick results")
        
        if not model:
            return "Error: Model not available"
            
        # Use Whisper's built-in ability to process portions
        # It naturally processes beginning segments first
        segments, info = model.transcribe(
            file_path,
            language=language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            no_speech_threshold=0.4,
            condition_on_previous_text=False,
            word_timestamps=False,
            vad_filter=True
        )
        
        # Collect only first portion for demo
        transcript_parts = []
        word_count = 0
        target_words = 200  # ~1-2 minutes of speech
        
        for segment in segments:
            text = segment.text.strip()
            if text:
                transcript_parts.append(text)
                word_count += len(text.split())
                
                # Stop after getting enough for demo
                if word_count >= target_words:
                    logger.warning(f"🎬 Demo portion complete: {word_count} words")
                    break
        
        transcript = " ".join(transcript_parts)
        return transcript + "\n\n[Demo Mode: First portion processed for quick demonstration]"
        
    except Exception as e:
        return f"Demo mode failed: {e}"

def transcribe_sample_segments(file_path: str, language="en", max_segments=50) -> str:
    """
    🚀 LIGHTNING DEMO: Process only first N segments for instant results
    """
    try:
        logger.warning(f"⚡ LIGHTNING MODE: Processing max {max_segments} segments")
        
        segments, info = model.transcribe(
            file_path,
            language=language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False,
            vad_filter=True
        )
        
        # Take only first N segments
        transcript_parts = []
        count = 0
        
        for segment in segments:
            if count >= max_segments:
                break
                
            text = segment.text.strip()
            if text:
                transcript_parts.append(text)
                count += 1
        
        transcript = " ".join(transcript_parts)
        logger.warning(f"⚡ Lightning complete: {count} segments processed")
        return transcript + f"\n\n[Lightning Mode: First {count} segments processed in seconds]"
        
    except Exception as e:
        return f"Lightning mode failed: {e}"

def get_model_info():
    """Get model information"""
    return {
        "model": "tiny.en (English-only)",
        "requirements": "NO FFmpeg needed!",
        "mode": "Direct file processing",
        "cooling": "Built-in breaks",
        "demo_functions": [
            "transcribe_first_portion()",
            "transcribe_sample_segments()",
            "transcribe_audio()"
        ]
    }

# Quick test function
def test_model():
    """Test if everything is working"""
    if model:
        return "✅ Model loaded successfully - ready to transcribe!"
    else:
        return "❌ Model failed to load - check faster-whisper installation"