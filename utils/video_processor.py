import cv2
import os
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from .video_processing import process_frame_with_green_screen
from .blur_processing import process_blur_frame
from .text_rendering import smart_text_wrap, render_text_with_emoji_multiline
from .file_operations import (add_audio_to_video, add_background_music_to_video, add_dual_audio_to_video,
                           get_video_properties, get_audio_files, create_output_folder,
                           get_video_files, get_gif_files, get_all_media_files, is_gif_file)
from .green_screen_detection import create_green_screen_mask
from .gif_processing import (process_gif_greenscreen, process_gif_blur, extract_gif_frames, 
                           process_video_with_gif_template)
from .narasi_processing import process_narasi_mode
from .gpu_config import gpu_config
import threading

class VideoProcessor:
    def __init__(self, gui):
        self.gui = gui
        self.FRAME_WIDTH = 1080
        self.FRAME_HEIGHT = 1920
    
    def get_font_file(self, font_name):
        """Mendapatkan file font berdasarkan nama font."""
        font_mapping = {
            "Arial": "arial.ttf",
            "Times New Roman": "times.ttf",
            "Helvetica": "arial.ttf",  # fallback
            "Courier New": "cour.ttf",
            "Verdana": "verdana.ttf",
            "Georgia": "georgia.ttf",
            "Comic Sans MS": "comic.ttf",
            "Impact": "impact.ttf",
            "Trebuchet MS": "trebuc.ttf",
            "Tahoma": "tahoma.ttf"
        }
        return font_mapping.get(font_name, "arial.ttf")
    
    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def get_random_audio_file(self, audio_folder):
        """Mendapatkan file audio random dari folder."""
        if not audio_folder or not os.path.exists(audio_folder):
            return None
        
        audio_files = get_audio_files(audio_folder)
        if not audio_files:
            return None
        
        return os.path.join(audio_folder, random.choice(audio_files))
    
    def get_template_for_processing(self, template_path):
        """Get template for processing - handles both static images and GIFs."""
        if template_path.lower().endswith('.gif'):
            # For GIF templates, extract first frame as the base template
            frames, _ = extract_gif_frames(template_path)
            if frames:
                return frames[0]  # Use first frame as template
            else:
                raise Exception("Could not extract frames from GIF template")
        else:
            # Static image template
            template = cv2.imread(template_path)
            if template is None:
                raise Exception("Could not load image template")
            return template
    
    def get_gpu_video_writer(self, output_path, fps, gpu_settings):
        """Membuat VideoWriter dengan GPU acceleration jika tersedia."""
        # Get safe fourcc codes
        fourcc_options = gpu_config.get_safe_fourcc_codes()
        
        # Try each fourcc code until one works
        for fourcc_name, fourcc_code in fourcc_options:
            try:
                out = cv2.VideoWriter(output_path, fourcc_code, fps, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
                
                if out.isOpened():
                    print(f"✅ VideoWriter created with {fourcc_name} codec")
                    return out
                else:
                    out.release()
            except Exception as e:
                print(f"⚠️ Failed to create VideoWriter with {fourcc_name}: {e}")
                continue
        
        # Final fallback - try with -1 (let OpenCV choose)
        try:
            print("🔄 Using OpenCV auto-selection for codec")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            if out.isOpened():
                print("✅ VideoWriter created with auto-selected codec")
                return out
            else:
                out.release()
        except Exception as e:
            print(f"❌ All VideoWriter attempts failed: {e}")
        
        # Last resort - create a dummy writer
        print("🆘 Creating fallback VideoWriter")
        return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
    
    def get_gpu_video_capture(self, video_path, gpu_settings):
        """Membuat VideoCapture dengan GPU acceleration jika tersedia."""
        # Always use standard VideoCapture for compatibility
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                if gpu_settings['enabled'] and gpu_settings['decoder']:
                    print(f"✅ VideoCapture with GPU decoder: {gpu_settings['decoder']}")
                else:
                    print("✅ VideoCapture with CPU decoder")
                return cap
            else:
                cap.release()
        except Exception as e:
            print(f"⚠️ VideoCapture failed: {e}")
        
        # Fallback
        print("🔄 Using fallback VideoCapture")
        return cv2.VideoCapture(video_path)
    
    def optimize_opencv_performance(self):
        """Optimize OpenCV performance settings."""
        try:
            # Enable OpenCL if available
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                print("🚀 OpenCL acceleration enabled")
            
            # Set number of threads for OpenCV
            cv2.setNumThreads(cv2.getNumberOfCPUs())
            print(f"🔧 OpenCV threads set to: {cv2.getNumberOfCPUs()}")
            
        except Exception as e:
            print(f"⚠️ Performance optimization failed: {e}")
    
    def process_single_video_greenscreen(self, video_path, template_path, template_mask, output_path, text_settings, audio_settings, gpu_settings):
        """Memproses satu video dengan mode green screen."""
        # Optimize performance
        self.optimize_opencv_performance()
        
        # Check if template is a GIF - IMPORTANT: Output is still MP4!
        if template_path.lower().endswith('.gif'):
            print(f"🎬 Processing video with animated GIF template -> MP4 output")
            # Use special GIF processing function that outputs MP4
            temp_output = output_path.replace('.mp4', '_temp.mp4')
            success = process_video_with_gif_template(template_path, video_path, temp_output, text_settings)
            
            if success:
                # Handle audio processing
                self.handle_audio_processing(temp_output, video_path, output_path, audio_settings)
                return True
            else:
                return False
        
        # Regular static template processing
        template = self.get_template_for_processing(template_path)
        template = cv2.resize(template, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        
        cap = self.get_gpu_video_capture(video_path, gpu_settings)
        fps, _, _ = get_video_properties(video_path)
        
        temp_output = output_path.replace('.mp4', '_temp.mp4')
        out = self.get_gpu_video_writer(temp_output, fps, gpu_settings)
        
        video_name = os.path.basename(video_path)
        frame_count = 0
        
        print(f"🎬 Processing {video_name} with GPU: {'Enabled' if gpu_settings['enabled'] else 'Disabled'}")
        
        try:
            while True:
                ret, video_frame = cap.read()
                if not ret:
                    break
                
                processed_frame = process_frame_with_green_screen(template, video_frame, template_mask)
                
                if text_settings['enabled']:
                    processed_frame = self.add_text_overlay(processed_frame, video_name, text_settings)
                
                # Ensure frame is the correct size
                if processed_frame.shape[:2] != (self.FRAME_HEIGHT, self.FRAME_WIDTH):
                    processed_frame = cv2.resize(processed_frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
                
                out.write(processed_frame)
                frame_count += 1
                
                # Progress update every 30 frames
                if frame_count % 30 == 0:
                    print(f"📊 Processed {frame_count} frames")
        
        except Exception as e:
            print(f"❌ Error during video processing: {e}")
        
        finally:
            cap.release()
            out.release()
            print(f"✅ Video processing completed: {frame_count} frames")
        
        # Handle audio processing
        self.handle_audio_processing(temp_output, video_path, output_path, audio_settings)
        return True
    
    def process_single_video_blur(self, video_path, output_path, blur_settings, text_settings, audio_settings, gpu_settings):
        """Memproses satu video dengan mode blur background."""
        # Optimize performance
        self.optimize_opencv_performance()
        
        cap = self.get_gpu_video_capture(video_path, gpu_settings)
        fps, _, _ = get_video_properties(video_path)
        
        temp_output = output_path.replace('.mp4', '_temp.mp4')
        out = self.get_gpu_video_writer(temp_output, fps, gpu_settings)
        
        video_name = os.path.basename(video_path)
        frame_count = 0
        
        print(f"🌀 Processing {video_name} with GPU: {'Enabled' if gpu_settings['enabled'] else 'Disabled'}")
        
        try:
            while True:
                ret, video_frame = cap.read()
                if not ret:
                    break
                
                processed_frame = process_blur_frame(
                    video_frame, 
                    blur_settings['crop_top'], 
                    blur_settings['crop_bottom'],
                    blur_settings['video_x_position'],
                    blur_settings['video_y_position'],
                    self.FRAME_WIDTH, 
                    self.FRAME_HEIGHT
                )
                
                # Add text overlay for blur mode
                if text_settings and text_settings['enabled']:
                    processed_frame = self.add_text_overlay(processed_frame, video_name, text_settings)
                
                # Ensure frame is the correct size
                if processed_frame.shape[:2] != (self.FRAME_HEIGHT, self.FRAME_WIDTH):
                    processed_frame = cv2.resize(processed_frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
                
                out.write(processed_frame)
                frame_count += 1
                
                # Progress update every 30 frames
                if frame_count % 30 == 0:
                    print(f"📊 Processed {frame_count} frames")
        
        except Exception as e:
            print(f"❌ Error during video processing: {e}")
        
        finally:
            cap.release()
            out.release()
            print(f"✅ Video processing completed: {frame_count} frames")
        
        # Handle audio processing
        self.handle_audio_processing(temp_output, video_path, output_path, audio_settings)
        return True
    
    def process_single_gif_greenscreen(self, gif_path, template_path, template_mask, output_path, text_settings):
        """Memproses satu GIF dengan mode green screen."""
        gif_name = os.path.basename(gif_path)
        print(f"🎬 Processing GIF: {gif_name}")
        
        try:
            # Get template (handles both static images and GIFs)
            template = self.get_template_for_processing(template_path)
            template = cv2.resize(template, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            
            success = process_gif_greenscreen(gif_path, template, template_mask, output_path, text_settings)
            if success:
                print(f"✅ GIF processing completed: {gif_name}")
            else:
                print(f"❌ GIF processing failed: {gif_name}")
            return success
        except Exception as e:
            print(f"❌ Error processing GIF {gif_name}: {e}")
            return False
    
    def process_single_gif_blur(self, gif_path, output_path, blur_settings, text_settings):
        """Memproses satu GIF dengan mode blur background."""
        gif_name = os.path.basename(gif_path)
        print(f"🌀 Processing GIF: {gif_name}")
        
        try:
            success = process_gif_blur(gif_path, output_path, blur_settings, text_settings)
            if success:
                print(f"✅ GIF processing completed: {gif_name}")
            else:
                print(f"❌ GIF processing failed: {gif_name}")
            return success
        except Exception as e:
            print(f"❌ Error processing GIF {gif_name}: {e}")
            return False
    
    def handle_audio_processing(self, temp_output, video_path, output_path, audio_settings):
        """Enhanced audio processing with dual audio support."""
        try:
            # Check if dual audio mixing is enabled
            if audio_settings.get('dual_audio_enabled', False):
                # Dual audio mixing mode
                if audio_settings['folder_path']:
                    background_audio_path = self.get_random_audio_file(audio_settings['folder_path'])
                    
                    if background_audio_path:
                        print(f"🎭 Dual audio mixing: Original + Background")
                        print(f"   Background music: {os.path.basename(background_audio_path)}")
                        
                        success = add_dual_audio_to_video(
                            temp_output, video_path, background_audio_path, output_path,
                            original_volume=audio_settings['original_volume'],
                            background_volume=audio_settings['background_volume']
                        )
                        
                        if success:
                            print(f"✅ Dual audio mixing completed")
                            return
                        else:
                            print(f"❌ Dual audio mixing failed, falling back to original audio")
                    else:
                        print(f"⚠️ No background music found for dual audio, using original audio")
                else:
                    print(f"⚠️ No background music folder selected for dual audio, using original audio")
                
                # Fallback to original audio if dual audio fails
                add_audio_to_video(temp_output, video_path, output_path)
                
            elif audio_settings['enabled'] and audio_settings['folder_path']:
                # Background music only mode
                background_audio_path = self.get_random_audio_file(audio_settings['folder_path'])
                
                if background_audio_path:
                    print(f"🎵 Adding background music: {os.path.basename(background_audio_path)}")
                    # Add background music with specified volume
                    success = add_background_music_to_video(
                        temp_output, video_path, background_audio_path, output_path,
                        volume=audio_settings['background_volume']
                    )
                    if not success:
                        print("🔄 Background music failed, using original audio")
                        add_audio_to_video(temp_output, video_path, output_path)
                else:
                    # Fallback to original audio if no background music found
                    print("🔄 No background music found, using original audio")
                    add_audio_to_video(temp_output, video_path, output_path)
            else:
                # Use original audio only
                print("🎵 Using original audio")
                add_audio_to_video(temp_output, video_path, output_path)
        
        except Exception as e:
            print(f"❌ Audio processing error: {e}")
            # Try to at least copy the temp file
            try:
                if os.path.exists(temp_output):
                    import shutil
                    shutil.copy2(temp_output, output_path)
                    print("🔄 Copied video without audio processing")
            except Exception as copy_error:
                print(f"❌ Failed to copy video: {copy_error}")
    
    def add_text_overlay(self, frame, video_name, text_settings):
        """Menambahkan overlay teks ke frame dengan auto-wrapping dan custom color."""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            try:
                font_file = self.get_font_file(text_settings['font'])
                font = ImageFont.truetype(font_file, text_settings['size'])
            except:
                font = ImageFont.load_default()
            
            video_name_text = os.path.splitext(video_name)[0].replace("_", " ")
            
            # Hitung area yang tersedia untuk text
            max_text_width = self.FRAME_WIDTH - 80  # 40px margin kiri-kanan
            
            # Auto-wrap text berdasarkan lebar frame
            lines = smart_text_wrap(video_name_text, draw, font, max_text_width, emoji_size=80)
            
            # Hitung posisi berdasarkan pengaturan
            x_percent = text_settings['x_position'] / 100
            y_percent = text_settings['y_position'] / 100
            
            # Hitung tinggi total text
            line_height = text_settings['size'] + 10
            total_text_height = len(lines) * line_height
            
            # Posisi Y berdasarkan persentase, dengan auto-adjustment
            base_y = int(y_percent * (self.FRAME_HEIGHT - total_text_height - 40))
            base_y = max(20, min(base_y, self.FRAME_HEIGHT - total_text_height - 20))
            
            # Render multiline text dengan emoji
            rendered_lines = render_text_with_emoji_multiline(
                draw, lines, font, self.FRAME_WIDTH, self.FRAME_HEIGHT, 
                base_y, emoji_size=80, line_spacing=10
            )
            
            # Get text color from settings (default to black if not specified)
            text_color = text_settings.get('color', '#000000')
            
            # Gambar text dan emoji
            for line_data in rendered_lines:
                for item_type, item, x_offset in line_data['items']:
                    if item_type == 'emoji':
                        # Posisi emoji disesuaikan dengan tinggi font
                        emoji_y = line_data['y'] + (text_settings['size'] - line_data['emoji_size']) // 2
                        pil_image.paste(item, (line_data['x_start'] + x_offset, emoji_y), item)
                    elif item_type == 'text':
                        # Gambar text dengan warna yang dipilih
                        draw.text((line_data['x_start'] + x_offset, line_data['y']), 
                                 item, font=font, fill=text_color)
            
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        except Exception as e:
            print(f"⚠️ Text overlay error: {e}")
            return frame  # Return original frame if text overlay fails
    
    def process_videos_bulk(self):
        """Memproses semua video dan GIF secara bulk dengan dukungan file tunggal dan narasi mode."""
        def process_thread():
            try:
                processing_mode = self.gui.get_processing_mode()
                gpu_settings = self.gui.get_gpu_settings()
                
                # Print GPU configuration
                print("\n🎮 GPU Configuration:")
                print(f"   GPU Available: {gpu_settings['config']['gpu_available']}")
                print(f"   GPU Enabled: {gpu_settings['enabled']}")
                print(f"   Encoder: {gpu_settings['encoder']}")
                print(f"   Decoder: {gpu_settings['decoder'] or 'CPU'}")
                if gpu_settings['config']['supported_encoders']:
                    print(f"   Supported Encoders: {', '.join(gpu_settings['config']['supported_encoders'])}")
                if gpu_settings['config']['supported_decoders']:
                    print(f"   Supported Decoders: {', '.join(gpu_settings['config']['supported_decoders'])}")
                print()
                
                # Get files to process (either from folder or selected files)
                files_to_process = self.gui.get_files_to_process()
                
                if not files_to_process:
                    self.gui.status_label.config(text="Pilih folder atau file terlebih dahulu!")
                    return
                
                # Special handling for Narasi Mode
                if processing_mode == "narasi":
                    self.process_narasi_mode(files_to_process, gpu_settings)
                    return
                
                # Regular processing for Green Screen and Blur modes
                # Validasi berdasarkan mode
                if processing_mode == "greenscreen":
                    if not self.gui.background_image_path:
                        self.gui.status_label.config(text="Pilih template terlebih dahulu!")
                        return
                
                # Setup berdasarkan mode
                if processing_mode == "greenscreen":
                    template_path = self.gui.background_image_path
                    
                    # Get template for mask creation (use first frame if GIF)
                    template_for_mask = self.get_template_for_processing(template_path)
                    template_for_mask = cv2.resize(template_for_mask, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
                    template_mask = create_green_screen_mask(template_for_mask)
                    
                    if np.sum(template_mask) == 0:
                        self.gui.status_label.config(text="Tidak ada green screen terdeteksi di template!")
                        return
                    
                    text_settings = self.gui.get_text_settings()
                    output_folder_name = "edited_videos_greenscreen"
                    
                    # Check if template is GIF
                    is_gif_template = template_path.lower().endswith('.gif')
                    print(f"🎬 Template type: {'Animated GIF' if is_gif_template else 'Static Image'}")
                    
                else:  # blur mode
                    template_path = None
                    template_mask = None
                    text_settings = self.gui.get_blur_text_settings()  # Get blur text settings
                    blur_settings = self.gui.get_blur_settings()
                    output_folder_name = "edited_videos_blur"
                
                # Dapatkan pengaturan audio
                audio_settings = self.gui.get_audio_settings()
                
                # Print audio configuration
                print("\n🎵 Audio Configuration:")
                print(f"   Background Music Enabled: {audio_settings['enabled']}")
                print(f"   Dual Audio Enabled: {audio_settings.get('dual_audio_enabled', False)}")
                if audio_settings.get('dual_audio_enabled', False):
                    print(f"   Original Audio Volume: {audio_settings['original_volume']}%")
                    print(f"   Background Music Volume: {audio_settings['background_volume']}%")
                elif audio_settings['enabled']:
                    print(f"   Background Music Volume: {audio_settings['background_volume']}%")
                print()
                
                # Validasi audio settings jika enabled
                if audio_settings['enabled'] or audio_settings.get('dual_audio_enabled', False):
                    if not audio_settings['folder_path']:
                        self.gui.status_label.config(text="Pilih folder audio terlebih dahulu!")
                        return
                    
                    audio_files = get_audio_files(audio_settings['folder_path'])
                    if not audio_files:
                        self.gui.status_label.config(text="Tidak ada file audio ditemukan di folder!")
                        return
                
                # Create output folder (use first file's directory as base)
                base_folder = files_to_process[0][0]  # Get directory from first file
                output_folder = create_output_folder(base_folder, output_folder_name)
                
                total_files = len(files_to_process)
                successful_files = 0
                
                # Count file types for logging
                video_count = sum(1 for folder_path, file_name in files_to_process if not is_gif_file(file_name))
                gif_count = sum(1 for folder_path, file_name in files_to_process if is_gif_file(file_name))
                
                print(f"📁 Processing {video_count} videos and {gif_count} GIFs")
                print(f"📂 Output folder: {output_folder}")
                
                for i, (folder_path, file_name) in enumerate(files_to_process):
                    try:
                        self.gui.update_progress(i, total_files, file_name)
                        
                        file_path = os.path.join(folder_path, file_name)
                        
                        # IMPORTANT: All outputs are MP4 (not GIF)
                        output_path = os.path.join(output_folder, f"edited_{os.path.splitext(file_name)[0]}.mp4")
                        
                        if is_gif_file(file_path):
                            print(f"🎬 Processing GIF input: {file_name} -> {os.path.basename(output_path)} (MP4)")
                        else:
                            print(f"🎬 Processing Video input: {file_name} -> {os.path.basename(output_path)} (MP4)")
                        
                        # Process based on file type and mode
                        if is_gif_file(file_path):
                            # Process GIF input -> MP4 output
                            if processing_mode == "greenscreen":
                                # Convert GIF processing to MP4 output
                                success = self.process_gif_to_mp4_greenscreen(
                                    file_path, template_path, template_mask, 
                                    output_path, text_settings, audio_settings, gpu_settings
                                )
                            else:  # blur mode
                                success = self.process_gif_to_mp4_blur(
                                    file_path, output_path, blur_settings, text_settings, audio_settings, gpu_settings
                                )
                            
                            if success:
                                successful_files += 1
                                print(f"✅ GIF -> MP4 processing successful: {file_name}")
                            else:
                                print(f"❌ GIF -> MP4 processing failed: {file_name}")
                        else:
                            # Process video input -> MP4 output
                            if processing_mode == "greenscreen":
                                self.process_single_video_greenscreen(
                                    file_path, template_path, template_mask, 
                                    output_path, text_settings, audio_settings, gpu_settings
                                )
                                successful_files += 1
                                print(f"✅ Video processing successful: {file_name}")
                            else:  # blur mode
                                self.process_single_video_blur(
                                    file_path, output_path, blur_settings, text_settings, audio_settings, gpu_settings
                                )
                                successful_files += 1
                                print(f"✅ Video processing successful: {file_name}")
                        
                    except Exception as e:
                        print(f"❌ Error processing {file_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                self.gui.update_progress(total_files, total_files, "Selesai!")
                
                # Enhanced status message with audio mode info
                mode_text = "Green Screen" if processing_mode == "greenscreen" else "Blur Background"
                gpu_info = f" (GPU: {'Enabled' if gpu_settings['enabled'] else 'Disabled'})"
                
                # Audio info with dual audio support
                audio_info = ""
                if audio_settings.get('dual_audio_enabled', False):
                    audio_info = f" (Dual Audio: Original {audio_settings['original_volume']}% + Background {audio_settings['background_volume']}%)"
                elif audio_settings['enabled']:
                    audio_info = f" (Background Music: {audio_settings['background_volume']}%)"
                else:
                    audio_info = " (Original Audio Only)"
                
                file_types = []
                if video_count > 0:
                    file_types.append(f"{video_count} videos")
                if gif_count > 0:
                    file_types.append(f"{gif_count} GIFs")
                
                file_info = " + ".join(file_types)
                
                # Add template info
                template_info = ""
                if processing_mode == "greenscreen" and template_path:
                    template_type = "GIF Template" if template_path.lower().endswith('.gif') else "Static Template"
                    template_info = f" with {template_type}"
                
                # Add text overlay info with color
                text_info = ""
                if processing_mode == "greenscreen" and text_settings['enabled']:
                    text_color = text_settings.get('color', '#000000')
                    text_info = f" + Text Overlay (Color: {text_color})"
                elif processing_mode == "blur" and text_settings['enabled']:
                    text_color = text_settings.get('color', '#000000')
                    text_info = f" + Text Overlay (Color: {text_color})"
                
                self.gui.status_label.config(
                    text=f"Selesai! {successful_files}/{total_files} files berhasil diproses ({file_info}) dengan mode {mode_text}{template_info}{text_info}{gpu_info}{audio_info}. Output: MP4 files di {output_folder}"
                )
                
            except Exception as e:
                print(f"❌ Processing error: {e}")
                import traceback
                traceback.print_exc()
                self.gui.status_label.config(text=f"Error: {str(e)}")
        
        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()
    
    def process_narasi_mode(self, files_to_process, gpu_settings):
        """Process narasi mode with video concatenation."""
        try:
            # Validasi narasi mode
            if not self.gui.background_image_path:
                self.gui.status_label.config(text="Pilih template terlebih dahulu!")
                return
            
            narasi_audio_path = self.gui.get_narasi_audio_path()
            if not narasi_audio_path:
                self.gui.status_label.config(text="Pilih file audio untuk Narasi Mode!")
                return
            
            # Get settings
            template_path = self.gui.background_image_path
            text_settings = self.gui.get_narasi_text_settings()
            
            # Prepare video paths
            video_paths = []
            for folder_path, file_name in files_to_process:
                file_path = os.path.join(folder_path, file_name)
                video_paths.append(file_path)
            
            print(f"🎙️ Narasi Mode: Processing {len(video_paths)} files")
            
            # Create output path
            base_folder = files_to_process[0][0]
            output_folder = create_output_folder(base_folder, "edited_videos_narasi")
            output_path = os.path.join(output_folder, "narasi_output.mp4")
            
            # Update progress
            self.gui.update_progress(0, 1, "Narasi Mode Processing...")
            
            # Process narasi mode
            success = process_narasi_mode(
                video_paths, template_path, narasi_audio_path, 
                output_path, text_settings, gpu_settings
            )
            
            if success:
                self.gui.update_progress(1, 1, "Selesai!")
                
                # Get file info
                video_count = sum(1 for path in video_paths if not is_gif_file(path))
                gif_count = sum(1 for path in video_paths if is_gif_file(path))
                
                file_types = []
                if video_count > 0:
                    file_types.append(f"{video_count} videos")
                if gif_count > 0:
                    file_types.append(f"{gif_count} GIFs")
                
                file_info = " + ".join(file_types)
                
                template_type = "GIF Template" if template_path.lower().endswith('.gif') else "Static Template"
                gpu_info = f" (GPU: {'Enabled' if gpu_settings['enabled'] else 'Disabled'})"
                
                # Add text color info
                text_info = ""
                if text_settings['enabled']:
                    text_color = text_settings.get('color', '#000000')
                    text_info = f" + Text Overlay (Color: {text_color})"
                
                self.gui.status_label.config(
                    text=f"Selesai! Narasi Mode berhasil: {len(video_paths)} files ({file_info}) digabung dengan {template_type}{text_info}{gpu_info}. Output: {output_path}"
                )
            else:
                self.gui.status_label.config(text="❌ Narasi Mode processing failed!")
                
        except Exception as e:
            print(f"❌ Narasi Mode error: {e}")
            import traceback
            traceback.print_exc()
            self.gui.status_label.config(text=f"Error Narasi Mode: {str(e)}")
    
    def process_gif_to_mp4_greenscreen(self, gif_path, template_path, template_mask, output_path, text_settings, audio_settings, gpu_settings):
        """Convert GIF input to MP4 output with greenscreen processing."""
        try:
            # Extract GIF frames
            frames, durations = extract_gif_frames(gif_path)
            if not frames:
                return False
            
            # Get template
            template = self.get_template_for_processing(template_path)
            template = cv2.resize(template, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            
            # Setup MP4 writer
            fps = 10  # Default FPS for GIF conversion
            temp_output = output_path.replace('.mp4', '_temp.mp4')
            out = self.get_gpu_video_writer(temp_output, fps, gpu_settings)
            
            gif_name = os.path.basename(gif_path)
            
            print(f"🎬 Converting GIF to MP4: {len(frames)} frames")
            
            for i, frame in enumerate(frames):
                # Process with greenscreen
                processed_frame = process_frame_with_green_screen(template, frame, template_mask)
                
                # Add text overlay
                if text_settings['enabled']:
                    processed_frame = self.add_text_overlay(processed_frame, gif_name, text_settings)
                
                # Ensure correct size
                if processed_frame.shape[:2] != (self.FRAME_HEIGHT, self.FRAME_WIDTH):
                    processed_frame = cv2.resize(processed_frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
                
                out.write(processed_frame)
                
                if (i + 1) % 10 == 0:
                    print(f"📊 Converted {i + 1}/{len(frames)} frames")
            
            out.release()
            
            # Handle audio (use silence for GIF conversion)
            self.handle_gif_audio_processing(temp_output, output_path, len(frames), fps, audio_settings)
            
            return True
            
        except Exception as e:
            print(f"❌ GIF to MP4 conversion error: {e}")
            return False
    
    def process_gif_to_mp4_blur(self, gif_path, output_path, blur_settings, text_settings, audio_settings, gpu_settings):
        """Convert GIF input to MP4 output with blur processing."""
        try:
            # Extract GIF frames
            frames, durations = extract_gif_frames(gif_path)
            if not frames:
                return False
            
            # Setup MP4 writer
            fps = 10  # Default FPS for GIF conversion
            temp_output = output_path.replace('.mp4', '_temp.mp4')
            out = self.get_gpu_video_writer(temp_output, fps, gpu_settings)
            
            gif_name = os.path.basename(gif_path)
            
            print(f"🌀 Converting GIF to MP4 with blur: {len(frames)} frames")
            
            for i, frame in enumerate(frames):
                # Process with blur
                processed_frame = process_blur_frame(
                    frame,
                    blur_settings['crop_top'],
                    blur_settings['crop_bottom'],
                    blur_settings['video_x_position'],
                    blur_settings['video_y_position'],
                    self.FRAME_WIDTH,
                    self.FRAME_HEIGHT
                )
                
                # Add text overlay for blur mode
                if text_settings and text_settings['enabled']:
                    processed_frame = self.add_text_overlay(processed_frame, gif_name, text_settings)
                
                # Ensure correct size
                if processed_frame.shape[:2] != (self.FRAME_HEIGHT, self.FRAME_WIDTH):
                    processed_frame = cv2.resize(processed_frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
                
                out.write(processed_frame)
                
                if (i + 1) % 10 == 0:
                    print(f"📊 Converted {i + 1}/{len(frames)} frames")
            
            out.release()
            
            # Handle audio (use silence for GIF conversion)
            self.handle_gif_audio_processing(temp_output, output_path, len(frames), fps, audio_settings)
            
            return True
            
        except Exception as e:
            print(f"❌ GIF to MP4 conversion error: {e}")
            return False
    
    def handle_gif_audio_processing(self, temp_output, output_path, frame_count, fps, audio_settings):
        """Handle audio for GIF to MP4 conversion with dual audio support."""
        try:
            # For GIF conversion, we only add background music (no original audio)
            if (audio_settings['enabled'] or audio_settings.get('dual_audio_enabled', False)) and audio_settings['folder_path']:
                # Add background music
                background_audio_path = self.get_random_audio_file(audio_settings['folder_path'])
                
                if background_audio_path:
                    print(f"🎵 Adding background music to converted MP4")
                    from moviepy.editor import VideoFileClip, AudioFileClip
                    
                    video_clip = VideoFileClip(temp_output)
                    background_audio = AudioFileClip(background_audio_path)
                    
                    # Adjust volume based on mode
                    if audio_settings.get('dual_audio_enabled', False):
                        # Use background volume from dual audio settings
                        background_volume = audio_settings['background_volume'] / 100.0
                    else:
                        # Use regular background volume
                        background_volume = audio_settings['background_volume'] / 100.0
                    
                    background_audio = background_audio.volumex(background_volume)
                    
                    # Loop or trim audio to match video duration
                    if background_audio.duration < video_clip.duration:
                        background_audio = background_audio.loop(duration=video_clip.duration)
                    else:
                        background_audio = background_audio.subclip(0, video_clip.duration)
                    
                    final_clip = video_clip.set_audio(background_audio)
                    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
                    
                    video_clip.close()
                    background_audio.close()
                    final_clip.close()
                    
                    # Remove temp file
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                    
                    return
            
            # No audio processing needed, just rename temp file
            if os.path.exists(temp_output):
                import shutil
                shutil.move(temp_output, output_path)
                
        except Exception as e:
            print(f"❌ Audio processing error for GIF conversion: {e}")
            # Fallback: just rename temp file
            if os.path.exists(temp_output):
                import shutil
                shutil.move(temp_output, output_path)