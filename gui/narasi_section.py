import tkinter as tk
from tkinter import filedialog
import os
from utils.file_operations import get_audio_files

class NarasiSection:
    """Narasi mode section of the GUI."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.narasi_audio_path = ""
        self.create_narasi_section()
    
    def create_narasi_section(self):
        """Create narasi mode section."""
        self.narasi_frame = tk.LabelFrame(
            self.parent_frame, 
            text="🎙️ Narasi Mode Settings", 
            font=("Arial", 11, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50", 
            padx=10, 
            pady=8
        )
        self.narasi_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Info
        info_label = tk.Label(
            self.narasi_frame, 
            text="Concatenate multiple videos and process with green screen template. Final duration matches audio duration.",
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0", 
            wraplength=600
        )
        info_label.pack(pady=5)
        
        # Audio Selection (Required)
        audio_section = tk.Frame(self.narasi_frame, bg="#f0f0f0")
        audio_section.pack(pady=8, fill=tk.X)
        
        audio_title = tk.Label(
            audio_section, 
            text="🎵 Audio File (Required)", 
            font=("Arial", 10, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50"
        )
        audio_title.pack(pady=(0, 5))
        
        self.audio_label = tk.Label(
            audio_section, 
            text="No audio file selected", 
            font=("Arial", 10), 
            bg="#f0f0f0", 
            fg="#7f8c8d"
        )
        self.audio_label.pack(pady=3)
        
        select_audio_btn = tk.Button(
            audio_section, 
            text="📁 Select Audio File", 
            command=self.select_audio_file, 
            font=("Arial", 10, "bold"), 
            bg="#e67e22", 
            fg="white", 
            activebackground="#d35400"
        )
        select_audio_btn.pack(pady=5)
        
        self.audio_info_label = tk.Label(
            audio_section, 
            text="", 
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0"
        )
        self.audio_info_label.pack(pady=2)
        
        # Processing Info
        process_info = tk.Frame(self.narasi_frame, bg="#f0f0f0")
        process_info.pack(pady=8, fill=tk.X)
        
        process_title = tk.Label(
            process_info, 
            text="⚙️ Processing Logic", 
            font=("Arial", 10, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50"
        )
        process_title.pack(pady=(0, 5))
        
        process_steps = tk.Label(
            process_info, 
            text="1. All selected videos will be concatenated into one video\n"
                 "2. Processed with green screen template + text overlay\n"
                 "3. Final duration will match the audio duration\n"
                 "4. If video is shorter than audio: video will loop\n"
                 "5. If video is longer than audio: video will be cut\n"
                 "6. Output: Single MP4 file with synchronized audio",
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0",
            justify=tk.LEFT
        )
        process_steps.pack(pady=2)
    
    def select_audio_file(self):
        """Select audio file for narasi mode."""
        audio_path = filedialog.askopenfilename(
            title="Select Audio File for Narasi Mode",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.aac *.m4a *.ogg *.flac *.wma"),
                ("MP3 Files", "*.mp3"),
                ("WAV Files", "*.wav"),
                ("All Files", "*.*")
            ]
        )
        
        if audio_path:
            self.narasi_audio_path = audio_path
            filename = os.path.basename(audio_path)
            self.audio_label.config(text=f"Audio: {filename}")
            
            # Get audio duration info
            try:
                from moviepy.editor import AudioFileClip
                audio_clip = AudioFileClip(audio_path)
                duration = audio_clip.duration
                audio_clip.close()
                
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                
                self.audio_info_label.config(
                    text=f"Duration: {minutes}:{seconds:02d} - Final video will match this duration"
                )
            except Exception as e:
                self.audio_info_label.config(text=f"Could not read audio duration: {str(e)}")
    
    def get_narasi_audio_path(self):
        """Get selected audio path."""
        return self.narasi_audio_path
    
    def pack_forget(self):
        """Hide narasi section."""
        if hasattr(self, 'narasi_frame'):
            self.narasi_frame.pack_forget()
    
    def pack(self, **kwargs):
        """Show narasi section."""
        if hasattr(self, 'narasi_frame'):
            self.narasi_frame.pack(**kwargs)