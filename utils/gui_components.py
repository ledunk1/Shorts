import tkinter as tk
from tkinter import filedialog, ttk, font
import os
import cv2
from .green_screen_detection import create_green_screen_mask
from .file_operations import get_video_files, get_audio_files, get_gif_files, is_gif_file
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from .text_rendering import smart_text_wrap, render_text_with_emoji_multiline
from .gpu_config import gpu_config

# Import modular GUI components
from gui.base_gui import BaseGUI
from gui.header_section import HeaderSection
from gui.mode_section import ModeSection
from gui.gpu_section import GPUSection
from gui.template_section import TemplateSection
from gui.narasi_section import NarasiSection

class VideoEditorGUI(BaseGUI):
    def __init__(self, root):
        super().__init__(root)
        
        # Initialize variables
        self.folder_path = ""
        self.audio_folder_path = ""
        self.selected_files = []  # For single file selection
        self.is_single_file_mode = False
        
        # Mode selection
        self.processing_mode = tk.StringVar(value="greenscreen")
        
        # Green screen mode settings
        self.text_enabled = tk.BooleanVar(value=True)
        self.text_x_position = tk.IntVar(value=50)
        self.text_y_position = tk.IntVar(value=10)
        self.text_size = tk.IntVar(value=65)
        self.selected_font = tk.StringVar(value="Arial")
        self.text_color = tk.StringVar(value="#000000")  # NEW: Text color for green screen
        
        # Blur mode settings (now includes text settings)
        self.crop_top = tk.IntVar(value=0)
        self.crop_bottom = tk.IntVar(value=0)
        self.video_x_position = tk.IntVar(value=50)
        self.video_y_position = tk.IntVar(value=50)
        
        # Blur mode text settings (separate from green screen)
        self.blur_text_enabled = tk.BooleanVar(value=True)
        self.blur_text_x_position = tk.IntVar(value=50)
        self.blur_text_y_position = tk.IntVar(value=10)
        self.blur_text_size = tk.IntVar(value=65)
        self.blur_selected_font = tk.StringVar(value="Arial")
        self.blur_text_color = tk.StringVar(value="#000000")  # NEW: Text color for blur mode
        
        # Narasi mode text settings (same as green screen)
        self.narasi_text_enabled = tk.BooleanVar(value=True)
        self.narasi_text_x_position = tk.IntVar(value=50)
        self.narasi_text_y_position = tk.IntVar(value=10)
        self.narasi_text_size = tk.IntVar(value=65)
        self.narasi_selected_font = tk.StringVar(value="Arial")
        self.narasi_text_color = tk.StringVar(value="#000000")  # NEW: Text color for narasi mode
        
        # Enhanced Audio settings with dual audio support
        self.audio_enabled = tk.BooleanVar(value=False)
        self.dual_audio_enabled = tk.BooleanVar(value=False)  # NEW: Dual audio mixing
        self.original_audio_volume = tk.IntVar(value=100)     # NEW: Original audio volume
        self.background_audio_volume = tk.IntVar(value=50)    # NEW: Background audio volume (renamed)
        
        # GPU settings
        self.gpu_enabled = tk.BooleanVar(value=gpu_config.USE_GPU)
        self.selected_encoder = tk.StringVar(value=gpu_config.get_optimal_encoder())
        self.selected_decoder = tk.StringVar(value=gpu_config.get_optimal_decoder() or "CPU")
        
        self.setup_gui()
    
    @property
    def background_image_path(self):
        """Get background image path from template section."""
        return self.template_section.background_image_path if hasattr(self, 'template_section') else ""
    
    def setup_gui(self):
        """Setup GUI components using modular approach with responsive design."""
        # Create all sections with consistent spacing
        self.header_section = HeaderSection(self.scrollable_frame)
        self.mode_section = ModeSection(self.scrollable_frame, self.processing_mode, self.on_mode_change)
        self.gpu_section = GPUSection(self.scrollable_frame, self.gpu_enabled, self.selected_encoder, self.selected_decoder)
        self.template_section = TemplateSection(self.scrollable_frame, self.update_preview)
        self.narasi_section = NarasiSection(self.scrollable_frame)
        
        self.create_text_settings_section()
        self.create_narasi_text_settings_section()
        self.create_blur_settings_section()
        self.create_audio_settings_section()
        self.create_folder_section()
        self.create_process_section()
        self.create_progress_section()
        self.create_instructions()
        self.create_footer()
        
        # Initialize mode display
        self.on_mode_change()
    
    def on_mode_change(self):
        """Handle mode change with responsive layout updates."""
        mode = self.processing_mode.get()
        self.mode_section.update_description(mode)
        
        if mode == "greenscreen":
            self.show_greenscreen_settings()
            self.hide_blur_settings()
            self.hide_narasi_settings()
        elif mode == "blur":
            self.hide_greenscreen_settings()
            self.show_blur_settings()
            self.hide_narasi_settings()
        elif mode == "narasi":
            self.show_narasi_settings()
            self.hide_blur_settings()
            # Template section is shared with narasi mode
        
        self.update_preview()
        
        # Update audio note visibility
        if hasattr(self, 'audio_note_label'):
            if mode == "narasi":
                self.audio_note_label.pack(pady=2, before=self.audio_folder_label)
            else:
                self.audio_note_label.pack_forget()
        
        # Update narasi note visibility
        if hasattr(self, 'narasi_note_label'):
            if mode == "narasi":
                self.narasi_note_label.pack(pady=2)
            else:
                self.narasi_note_label.pack_forget()
    
    def show_greenscreen_settings(self):
        """Show greenscreen settings."""
        self.template_section.pack(pady=10, padx=20, fill=tk.X, before=self.audio_frame)
        if hasattr(self, 'text_frame'):
            self.text_frame.pack(pady=10, padx=20, fill=tk.X, before=self.audio_frame)
    
    def hide_greenscreen_settings(self):
        """Hide greenscreen settings."""
        self.template_section.pack_forget()
        if hasattr(self, 'text_frame'):
            self.text_frame.pack_forget()
    
    def show_blur_settings(self):
        """Show blur settings."""
        if hasattr(self, 'blur_frame'):
            self.blur_frame.pack(pady=10, padx=20, fill=tk.X, before=self.audio_frame)
    
    def hide_blur_settings(self):
        """Hide blur settings."""
        if hasattr(self, 'blur_frame'):
            self.blur_frame.pack_forget()
    
    def show_narasi_settings(self):
        """Show narasi settings."""
        self.template_section.pack(pady=10, padx=20, fill=tk.X, before=self.audio_frame)
        self.narasi_section.pack(pady=10, padx=20, fill=tk.X, before=self.audio_frame)
        if hasattr(self, 'narasi_text_frame'):
            self.narasi_text_frame.pack(pady=10, padx=20, fill=tk.X, before=self.audio_frame)
    
    def hide_narasi_settings(self):
        """Hide narasi settings."""
        self.narasi_section.pack_forget()
        if hasattr(self, 'narasi_text_frame'):
            self.narasi_text_frame.pack_forget()
    
    def create_text_settings_section(self):
        """Create text settings section for green screen mode with responsive layout and color picker."""
        self.text_frame = self.create_label_frame("Green Screen Text Settings", "✏️")
        
        # Enable checkbox
        text_checkbox = tk.Checkbutton(
            self.text_frame, 
            text="📝 Enable Text Overlay", 
            variable=self.text_enabled, 
            font=("Arial", 10), 
            bg="#f0f0f0", 
            command=self.update_preview
        )
        text_checkbox.pack(anchor=tk.W, pady=5)
        
        # Controls in responsive grid
        controls_frame = tk.Frame(self.text_frame, bg="#f0f0f0")
        controls_frame.pack(pady=8, fill=tk.X)
        
        # Configure responsive grid (3 columns now)
        self.configure_responsive_grid(controls_frame, 3)
        
        # X Position
        x_frame = tk.Frame(controls_frame, bg="#f0f0f0")
        x_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        tk.Label(x_frame, text="📍 X Position:", font=("Arial", 9), bg="#f0f0f0").pack()
        x_scale = tk.Scale(
            x_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
            variable=self.text_x_position, command=self.on_position_change, length=200
        )
        x_scale.pack(fill=tk.X)
        
        # Y Position
        y_frame = tk.Frame(controls_frame, bg="#f0f0f0")
        y_frame.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        tk.Label(y_frame, text="📍 Y Position:", font=("Arial", 9), bg="#f0f0f0").pack()
        y_scale = tk.Scale(
            y_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
            variable=self.text_y_position, command=self.on_position_change, length=200
        )
        y_scale.pack(fill=tk.X)
        
        # Text Color
        color_frame = tk.Frame(controls_frame, bg="#f0f0f0")
        color_frame.grid(row=0, column=2, padx=10, pady=5, sticky="ew")
        tk.Label(color_frame, text="🎨 Text Color:", font=("Arial", 9), bg="#f0f0f0").pack()
        
        color_button_frame = tk.Frame(color_frame, bg="#f0f0f0")
        color_button_frame.pack(pady=2)
        
        self.text_color_button = tk.Button(
            color_button_frame,
            text="⬛",
            bg=self.text_color.get(),
            width=3,
            height=1,
            command=lambda: self.choose_color(self.text_color, self.text_color_button, self.update_preview),
            relief="solid",
            borderwidth=1
        )
        self.text_color_button.pack(side=tk.LEFT, padx=2)
        
        self.text_color_label = tk.Label(
            color_button_frame,
            text=self.text_color.get(),
            font=("Arial", 8),
            bg="#f0f0f0"
        )
        self.text_color_label.pack(side=tk.LEFT, padx=5)
        
        # Text Size
        size_frame = tk.Frame(controls_frame, bg="#f0f0f0")
        size_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        tk.Label(size_frame, text="📏 Text Size:", font=("Arial", 9), bg="#f0f0f0").pack()
        size_scale = tk.Scale(
            size_frame, from_=20, to=120, orient=tk.HORIZONTAL, 
            variable=self.text_size, command=self.on_position_change, length=200
        )
        size_scale.pack(fill=tk.X)
        
        # Font
        font_frame = tk.Frame(controls_frame, bg="#f0f0f0")
        font_frame.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        tk.Label(font_frame, text="🎨 Font:", font=("Arial", 9), bg="#f0f0f0").pack()
        
        available_fonts = sorted(font.families())
        common_fonts = ["Arial", "Times New Roman", "Helvetica", "Courier New", "Verdana"]
        font_list = common_fonts + [f for f in available_fonts if f not in common_fonts]
        
        self.font_combobox = ttk.Combobox(
            font_frame, textvariable=self.selected_font, 
            values=font_list, state="readonly", width=18
        )
        self.font_combobox.pack(pady=2, fill=tk.X)
        self.font_combobox.bind("<<ComboboxSelected>>", self.on_font_change)
    
    def create_narasi_text_settings_section(self):
        """Create text settings section for narasi mode with responsive layout and color picker."""
        self.narasi_text_frame = self.create_label_frame("Narasi Text Settings", "✏️")
        
        # Enable checkbox
        narasi_text_checkbox = tk.Checkbutton(
            self.narasi_text_frame, 
            text="📝 Enable Text Overlay", 
            variable=self.narasi_text_enabled, 
            font=("Arial", 10), 
            bg="#f0f0f0", 
            command=self.update_preview
        )
        narasi_text_checkbox.pack(anchor=tk.W, pady=5)
        
        # Controls in responsive grid
        narasi_controls_frame = tk.Frame(self.narasi_text_frame, bg="#f0f0f0")
        narasi_controls_frame.pack(pady=8, fill=tk.X)
        
        # Configure responsive grid (3 columns)
        self.configure_responsive_grid(narasi_controls_frame, 3)
        
        # X Position
        narasi_x_frame = tk.Frame(narasi_controls_frame, bg="#f0f0f0")
        narasi_x_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        tk.Label(narasi_x_frame, text="📍 X Position:", font=("Arial", 9), bg="#f0f0f0").pack()
        narasi_x_scale = tk.Scale(
            narasi_x_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
            variable=self.narasi_text_x_position, command=self.on_narasi_position_change, length=200
        )
        narasi_x_scale.pack(fill=tk.X)
        
        # Y Position
        narasi_y_frame = tk.Frame(narasi_controls_frame, bg="#f0f0f0")
        narasi_y_frame.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        tk.Label(narasi_y_frame, text="📍 Y Position:", font=("Arial", 9), bg="#f0f0f0").pack()
        narasi_y_scale = tk.Scale(
            narasi_y_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
            variable=self.narasi_text_y_position, command=self.on_narasi_position_change, length=200
        )
        narasi_y_scale.pack(fill=tk.X)
        
        # Text Color
        narasi_color_frame = tk.Frame(narasi_controls_frame, bg="#f0f0f0")
        narasi_color_frame.grid(row=0, column=2, padx=10, pady=5, sticky="ew")
        tk.Label(narasi_color_frame, text="🎨 Text Color:", font=("Arial", 9), bg="#f0f0f0").pack()
        
        narasi_color_button_frame = tk.Frame(narasi_color_frame, bg="#f0f0f0")
        narasi_color_button_frame.pack(pady=2)
        
        self.narasi_text_color_button = tk.Button(
            narasi_color_button_frame,
            text="⬛",
            bg=self.narasi_text_color.get(),
            width=3,
            height=1,
            command=lambda: self.choose_color(self.narasi_text_color, self.narasi_text_color_button, self.update_preview),
            relief="solid",
            borderwidth=1
        )
        self.narasi_text_color_button.pack(side=tk.LEFT, padx=2)
        
        self.narasi_text_color_label = tk.Label(
            narasi_color_button_frame,
            text=self.narasi_text_color.get(),
            font=("Arial", 8),
            bg="#f0f0f0"
        )
        self.narasi_text_color_label.pack(side=tk.LEFT, padx=5)
        
        # Text Size
        narasi_size_frame = tk.Frame(narasi_controls_frame, bg="#f0f0f0")
        narasi_size_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        tk.Label(narasi_size_frame, text="📏 Text Size:", font=("Arial", 9), bg="#f0f0f0").pack()
        narasi_size_scale = tk.Scale(
            narasi_size_frame, from_=20, to=120, orient=tk.HORIZONTAL, 
            variable=self.narasi_text_size, command=self.on_narasi_position_change, length=200
        )
        narasi_size_scale.pack(fill=tk.X)
        
        # Font
        narasi_font_frame = tk.Frame(narasi_controls_frame, bg="#f0f0f0")
        narasi_font_frame.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        tk.Label(narasi_font_frame, text="🎨 Font:", font=("Arial", 9), bg="#f0f0f0").pack()
        
        available_fonts = sorted(font.families())
        common_fonts = ["Arial", "Times New Roman", "Helvetica", "Courier New", "Verdana"]
        font_list = common_fonts + [f for f in available_fonts if f not in common_fonts]
        
        self.narasi_font_combobox = ttk.Combobox(
            narasi_font_frame, textvariable=self.narasi_selected_font, 
            values=font_list, state="readonly", width=18
        )
        self.narasi_font_combobox.pack(pady=2, fill=tk.X)
        self.narasi_font_combobox.bind("<<ComboboxSelected>>", self.on_narasi_font_change)
    
    def create_blur_settings_section(self):
        """Create blur settings section with text settings and responsive layout."""
        self.blur_frame = self.create_label_frame("Blur Mode Settings", "🌀")
        
        info_label = tk.Label(
            self.blur_frame, 
            text="Create blurred background with cropped video overlay in 9:16 aspect ratio + text overlay (Supports GIF)",
            font=("Arial", 9), fg="#7f8c8d", bg="#f0f0f0", wraplength=800
        )
        info_label.pack(pady=5)
        
        # Video Position Controls
        video_controls_frame = tk.Frame(self.blur_frame, bg="#f0f0f0")
        video_controls_frame.pack(pady=8, fill=tk.X)
        
        video_title = tk.Label(
            video_controls_frame, 
            text="📹 Video Position & Crop Settings", 
            font=("Arial", 10, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50"
        )
        video_title.pack(pady=(0, 5))
        
        # Controls in responsive grid
        controls_frame = tk.Frame(video_controls_frame, bg="#f0f0f0")
        controls_frame.pack(pady=5, fill=tk.X)
        
        # Configure responsive grid
        self.configure_responsive_grid(controls_frame, 2)
        
        # Crop Top
        crop_top_frame = tk.Frame(controls_frame, bg="#f0f0f0")
        crop_top_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        tk.Label(crop_top_frame, text="✂️ Crop Top (%):", font=("Arial", 9), bg="#f0f0f0").pack()
        crop_top_scale = tk.Scale(
            crop_top_frame, from_=0, to=40, orient=tk.HORIZONTAL, 
            variable=self.crop_top, command=self.on_crop_change, length=200
        )
        crop_top_scale.pack(fill=tk.X)
        
        # Crop Bottom
        crop_bottom_frame = tk.Frame(controls_frame, bg="#f0f0f0")
        crop_bottom_frame.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        tk.Label(crop_bottom_frame, text="✂️ Crop Bottom (%):", font=("Arial", 9), bg="#f0f0f0").pack()
        crop_bottom_scale = tk.Scale(
            crop_bottom_frame, from_=0, to=40, orient=tk.HORIZONTAL, 
            variable=self.crop_bottom, command=self.on_crop_change, length=200
        )
        crop_bottom_scale.pack(fill=tk.X)
        
        # Video X Position
        video_x_frame = tk.Frame(controls_frame, bg="#f0f0f0")
        video_x_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        tk.Label(video_x_frame, text="↔️ Video X (%):", font=("Arial", 9), bg="#f0f0f0").pack()
        video_x_scale = tk.Scale(
            video_x_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
            variable=self.video_x_position, command=self.on_crop_change, length=200
        )
        video_x_scale.pack(fill=tk.X)
        
        # Video Y Position
        video_y_frame = tk.Frame(controls_frame, bg="#f0f0f0")
        video_y_frame.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        tk.Label(video_y_frame, text="↕️ Video Y (%):", font=("Arial", 9), bg="#f0f0f0").pack()
        video_y_scale = tk.Scale(
            video_y_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
            variable=self.video_y_position, command=self.on_crop_change, length=200
        )
        video_y_scale.pack(fill=tk.X)
        
        # Text Settings for Blur Mode
        self.create_blur_text_settings()
        
        # Preview
        blur_preview_frame = tk.Frame(self.blur_frame, bg="#e0e0e0", relief=tk.SUNKEN, bd=1)
        blur_preview_frame.pack(pady=8)
        
        blur_preview_title = tk.Label(
            blur_preview_frame, 
            text="📱 Blur Preview (9:16)", 
            font=("Arial", 10, "bold"), 
            bg="#e0e0e0"
        )
        blur_preview_title.pack(pady=5)
        
        self.blur_preview_label = tk.Label(
            blur_preview_frame, 
            text="Select video folder to see preview", 
            bg="#ffffff", 
            fg="#95a5a6", 
            width=22, 
            height=14,
            relief=tk.SUNKEN, 
            bd=1
        )
        self.blur_preview_label.pack(pady=8, padx=8)
    
    def create_blur_text_settings(self):
        """Create text settings for blur mode with responsive layout and color picker."""
        # Text Settings Section
        text_settings_frame = tk.Frame(self.blur_frame, bg="#f0f0f0")
        text_settings_frame.pack(pady=8, fill=tk.X)
        
        text_title = tk.Label(
            text_settings_frame, 
            text="✏️ Text Overlay Settings", 
            font=("Arial", 10, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50"
        )
        text_title.pack(pady=(0, 5))
        
        # Enable checkbox
        blur_text_checkbox = tk.Checkbutton(
            text_settings_frame, 
            text="📝 Enable Text Overlay", 
            variable=self.blur_text_enabled, 
            font=("Arial", 10), 
            bg="#f0f0f0", 
            command=self.on_blur_text_change
        )
        blur_text_checkbox.pack(anchor=tk.W, pady=5)
        
        # Text controls in responsive grid
        blur_text_controls_frame = tk.Frame(text_settings_frame, bg="#f0f0f0")
        blur_text_controls_frame.pack(pady=5, fill=tk.X)
        
        # Configure responsive grid (3 columns)
        self.configure_responsive_grid(blur_text_controls_frame, 3)
        
        # X Position
        blur_text_x_frame = tk.Frame(blur_text_controls_frame, bg="#f0f0f0")
        blur_text_x_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        tk.Label(blur_text_x_frame, text="📍 Text X Position:", font=("Arial", 9), bg="#f0f0f0").pack()
        blur_text_x_scale = tk.Scale(
            blur_text_x_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
            variable=self.blur_text_x_position, command=self.on_blur_text_change, length=200
        )
        blur_text_x_scale.pack(fill=tk.X)
        
        # Y Position
        blur_text_y_frame = tk.Frame(blur_text_controls_frame, bg="#f0f0f0")
        blur_text_y_frame.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        tk.Label(blur_text_y_frame, text="📍 Text Y Position:", font=("Arial", 9), bg="#f0f0f0").pack()
        blur_text_y_scale = tk.Scale(
            blur_text_y_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
            variable=self.blur_text_y_position, command=self.on_blur_text_change, length=200
        )
        blur_text_y_scale.pack(fill=tk.X)
        
        # Text Color
        blur_color_frame = tk.Frame(blur_text_controls_frame, bg="#f0f0f0")
        blur_color_frame.grid(row=0, column=2, padx=10, pady=5, sticky="ew")
        tk.Label(blur_color_frame, text="🎨 Text Color:", font=("Arial", 9), bg="#f0f0f0").pack()
        
        blur_color_button_frame = tk.Frame(blur_color_frame, bg="#f0f0f0")
        blur_color_button_frame.pack(pady=2)
        
        self.blur_text_color_button = tk.Button(
            blur_color_button_frame,
            text="⬛",
            bg=self.blur_text_color.get(),
            width=3,
            height=1,
            command=lambda: self.choose_color(self.blur_text_color, self.blur_text_color_button, self.on_blur_text_change),
            relief="solid",
            borderwidth=1
        )
        self.blur_text_color_button.pack(side=tk.LEFT, padx=2)
        
        self.blur_text_color_label = tk.Label(
            blur_color_button_frame,
            text=self.blur_text_color.get(),
            font=("Arial", 8),
            bg="#f0f0f0"
        )
        self.blur_text_color_label.pack(side=tk.LEFT, padx=5)
        
        # Text Size
        blur_text_size_frame = tk.Frame(blur_text_controls_frame, bg="#f0f0f0")
        blur_text_size_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        tk.Label(blur_text_size_frame, text="📏 Text Size:", font=("Arial", 9), bg="#f0f0f0").pack()
        blur_text_size_scale = tk.Scale(
            blur_text_size_frame, from_=20, to=120, orient=tk.HORIZONTAL, 
            variable=self.blur_text_size, command=self.on_blur_text_change, length=200
        )
        blur_text_size_scale.pack(fill=tk.X)
        
        # Font
        blur_text_font_frame = tk.Frame(blur_text_controls_frame, bg="#f0f0f0")
        blur_text_font_frame.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        tk.Label(blur_text_font_frame, text="🎨 Font:", font=("Arial", 9), bg="#f0f0f0").pack()
        
        available_fonts = sorted(font.families())
        common_fonts = ["Arial", "Times New Roman", "Helvetica", "Courier New", "Verdana"]
        font_list = common_fonts + [f for f in available_fonts if f not in common_fonts]
        
        self.blur_font_combobox = ttk.Combobox(
            blur_text_font_frame, textvariable=self.blur_selected_font, 
            values=font_list, state="readonly", width=18
        )
        self.blur_font_combobox.pack(pady=2, fill=tk.X)
        self.blur_font_combobox.bind("<<ComboboxSelected>>", self.on_blur_font_change)
    
    def create_audio_settings_section(self):
        """Create enhanced audio settings section with dual audio support."""
        self.audio_frame = self.create_label_frame("Enhanced Audio Settings", "🎵")
        self.audio_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Audio Mode Selection
        audio_mode_frame = tk.Frame(self.audio_frame, bg="#f0f0f0")
        audio_mode_frame.pack(pady=8, fill=tk.X)
        
        mode_title = tk.Label(
            audio_mode_frame, 
            text="🎚️ Audio Mode Selection", 
            font=("Arial", 10, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50"
        )
        mode_title.pack(pady=(0, 5))
        
        # Radio buttons for audio modes
        mode_buttons_frame = tk.Frame(audio_mode_frame, bg="#f0f0f0")
        mode_buttons_frame.pack(pady=5)
        
        # Original Audio Only (default)
        original_only_radio = tk.Radiobutton(
            mode_buttons_frame,
            text="🎬 Original Audio Only",
            variable=self.audio_enabled,
            value=False,
            font=("Arial", 10),
            bg="#f0f0f0",
            command=self.on_audio_mode_change
        )
        original_only_radio.pack(anchor=tk.W, pady=2)
        
        # Background Music Only
        background_only_radio = tk.Radiobutton(
            mode_buttons_frame,
            text="🎶 Background Music Only",
            variable=self.audio_enabled,
            value=True,
            font=("Arial", 10),
            bg="#f0f0f0",
            command=self.on_audio_mode_change
        )
        background_only_radio.pack(anchor=tk.W, pady=2)
        
        # Dual Audio Mixing (NEW)
        dual_audio_checkbox = tk.Checkbutton(
            mode_buttons_frame,
            text="🎭 Enable Dual Audio Mixing (Original + Background)",
            variable=self.dual_audio_enabled,
            font=("Arial", 10, "bold"),
            bg="#f0f0f0",
            fg="#8e44ad",
            command=self.on_dual_audio_change
        )
        dual_audio_checkbox.pack(anchor=tk.W, pady=2)
        
        # Note for narasi mode
        self.audio_note_label = tk.Label(
            self.audio_frame, 
            text="Note: In Narasi Mode, audio is required and selected separately", 
            font=("Arial", 9, "italic"), 
            fg="#e67e22", 
            bg="#f0f0f0"
        )
        # Will be packed conditionally based on mode
        
        # Folder selection
        self.audio_folder_label = tk.Label(
            self.audio_frame, 
            text="No audio folder selected", 
            font=("Arial", 10), 
            bg="#f0f0f0", 
            fg="#7f8c8d"
        )
        self.audio_folder_label.pack(pady=3)
        
        self.select_audio_folder_btn = self.create_button(
            self.audio_frame, 
            "📁 Select Audio Folder", 
            self.select_audio_folder,
            bg_color="#e67e22",
            state=tk.DISABLED
        )
        self.select_audio_folder_btn.pack(pady=5)
        
        self.audio_count_label = tk.Label(
            self.audio_frame, 
            text="", 
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0"
        )
        self.audio_count_label.pack(pady=2)
        
        # Volume controls with responsive layout
        self.create_volume_controls()
        
        # Initialize audio mode
        self.on_audio_mode_change()
    
    def create_volume_controls(self):
        """Create volume control section with dual audio support."""
        volume_frame = tk.Frame(self.audio_frame, bg="#f0f0f0")
        volume_frame.pack(pady=8, fill=tk.X)
        
        volume_title = tk.Label(
            volume_frame, 
            text="🔊 Volume Controls", 
            font=("Arial", 10, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50"
        )
        volume_title.pack(pady=(0, 5))
        
        # Volume controls container
        volume_controls_frame = tk.Frame(volume_frame, bg="#f0f0f0")
        volume_controls_frame.pack(pady=5, fill=tk.X)
        
        # Configure responsive grid
        self.configure_responsive_grid(volume_controls_frame, 2)
        
        # Original Audio Volume (for dual audio mode)
        original_volume_frame = tk.Frame(volume_controls_frame, bg="#f0f0f0")
        original_volume_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        tk.Label(
            original_volume_frame, 
            text="🎬 Original Audio Volume:", 
            font=("Arial", 10), 
            bg="#f0f0f0"
        ).pack()
        
        self.original_volume_scale = tk.Scale(
            original_volume_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
            variable=self.original_audio_volume, length=300, state=tk.DISABLED
        )
        self.original_volume_scale.pack(fill=tk.X, pady=3)
        
        # Original volume labels
        original_labels = tk.Frame(original_volume_frame, bg="#f0f0f0")
        original_labels.pack(fill=tk.X)
        
        tk.Label(original_labels, text="🔇 0%", font=("Arial", 8), bg="#f0f0f0").pack(side=tk.LEFT)
        tk.Label(original_labels, text="🔊 50%", font=("Arial", 8), bg="#f0f0f0").pack()
        tk.Label(original_labels, text="🔊 100%", font=("Arial", 8), bg="#f0f0f0").pack(side=tk.RIGHT)
        
        # Background Music Volume
        background_volume_frame = tk.Frame(volume_controls_frame, bg="#f0f0f0")
        background_volume_frame.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        tk.Label(
            background_volume_frame, 
            text="🎵 Background Music Volume:", 
            font=("Arial", 10), 
            bg="#f0f0f0"
        ).pack()
        
        self.background_volume_scale = tk.Scale(
            background_volume_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
            variable=self.background_audio_volume, length=300, state=tk.DISABLED
        )
        self.background_volume_scale.pack(fill=tk.X, pady=3)
        
        # Background volume labels
        background_labels = tk.Frame(background_volume_frame, bg="#f0f0f0")
        background_labels.pack(fill=tk.X)
        
        tk.Label(background_labels, text="🔇 0%", font=("Arial", 8), bg="#f0f0f0").pack(side=tk.LEFT)
        tk.Label(background_labels, text="🔊 50%", font=("Arial", 8), bg="#f0f0f0").pack()
        tk.Label(background_labels, text="🔊 100%", font=("Arial", 8), bg="#f0f0f0").pack(side=tk.RIGHT)
    
    def create_folder_section(self):
        """Create folder/file selection section with responsive layout."""
        folder_frame = self.create_label_frame("Video/GIF Selection", "📂")
        folder_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Selection mode buttons with responsive layout
        mode_frame = tk.Frame(folder_frame, bg="#f0f0f0")
        mode_frame.pack(pady=5, fill=tk.X)
        
        # Center the buttons
        button_container = tk.Frame(mode_frame, bg="#f0f0f0")
        button_container.pack()
        
        folder_btn = self.create_button(
            button_container, 
            "📁 Select Folder", 
            self.select_folder,
            bg_color="#27ae60"
        )
        folder_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        files_btn = self.create_button(
            button_container, 
            "📄 Select Files", 
            self.select_files,
            bg_color="#3498db"
        )
        files_btn.pack(side=tk.LEFT)
        
        # Status display
        self.folder_label = tk.Label(
            folder_frame, 
            text="No folder or files selected", 
            font=("Arial", 10), 
            bg="#f0f0f0", 
            fg="#7f8c8d"
        )
        self.folder_label.pack(pady=5)
        
        self.video_count_label = tk.Label(
            folder_frame, 
            text="", 
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0"
        )
        self.video_count_label.pack(pady=3)
        
        # Narasi mode note
        self.narasi_note_label = tk.Label(
            folder_frame, 
            text="📝 Narasi Mode: All selected videos will be concatenated into one output file", 
            font=("Arial", 9, "italic"), 
            fg="#8e44ad", 
            bg="#f0f0f0"
        )
        # Will be packed conditionally based on mode
    
    def create_process_section(self):
        """Create process section with responsive design."""
        process_frame = tk.Frame(self.scrollable_frame, bg="#f0f0f0")
        process_frame.pack(pady=20)
        
        # Create button with enhanced styling
        self.process_btn = tk.Button(
            process_frame, 
            text="🚀 Start Processing", 
            command=None,  # Will be set by callback
            font=("Arial", 14, "bold"),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            relief="flat",
            borderwidth=0,
            padx=40,
            pady=15,
            cursor="hand2"
        )
        self.process_btn.pack()
    
    def create_progress_section(self):
        """Create progress section with responsive layout."""
        progress_frame = self.create_label_frame("Progress", "📊")
        progress_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=600, mode='determinate')
        self.progress_bar.pack(pady=5, fill=tk.X, padx=20)
        
        self.status_label = tk.Label(
            progress_frame, 
            text="Ready to process videos and GIFs", 
            font=("Arial", 10), 
            fg="#2c3e50", 
            bg="#f0f0f0",
            wraplength=800
        )
        self.status_label.pack(pady=5, padx=10)
    
    def create_instructions(self):
        """Create instructions section with responsive layout."""
        instructions_frame = self.create_label_frame("Instructions", "📋")
        instructions_frame.pack(pady=10, padx=20, fill=tk.X)
        
        instructions_text = """🎮 GPU Acceleration:
1. Enable GPU acceleration for faster processing
2. Select optimal encoder/decoder for your hardware

🎬 Green Screen Mode:
1. Select template image with green screen area
2. Configure text overlay settings (position, size, font, color)

🌀 Blur Mode:
1. Adjust crop settings for 9:16 aspect ratio
2. Set video position within the frame
3. Configure text overlay settings (position, size, font, color)

🎙️ Narasi Mode:
1. Select template with green screen area
2. Select audio file (required) - final duration will match audio
3. Configure text overlay settings (position, size, font, color)
4. All videos will be concatenated into one output

🎭 Enhanced Audio Features:
1. Original Audio Only: Keep original video audio
2. Background Music Only: Replace with random background music
3. Dual Audio Mixing: Mix original audio with background music
4. Adjust volume levels for both original and background audio

🚀 Processing:
1. Select folder containing videos/GIFs OR select individual files
2. Click 'Start Processing' and wait for completion

📁 Supported Formats:
• Videos: MP4, AVI, MOV, MKV, WMV, FLV
• GIFs: Animated GIF files
• Images: JPG, PNG, BMP (for templates)
• Audio: MP3, WAV, AAC, M4A, OGG, FLAC, WMA"""
        
        instructions_label = tk.Label(
            instructions_frame, 
            text=instructions_text,
            font=("Arial", 9), 
            bg="#f0f0f0", 
            fg="#2c3e50",
            justify=tk.LEFT,
            wraplength=900
        )
        instructions_label.pack(padx=10, pady=8)
    
    def create_footer(self):
        """Create footer with responsive design."""
        footer_frame = tk.Frame(self.scrollable_frame, bg="#f0f0f0")
        footer_frame.pack(pady=15)
        
        # Add separator
        separator = tk.Frame(footer_frame, height=1, bg="#bdc3c7")
        separator.pack(fill=tk.X, pady=(0, 10), padx=50)
        
        footer_label = tk.Label(
            footer_frame, 
            text="Created by Trialota - Enhanced with GPU Acceleration, Audio Features, GIF Support, Narasi Mode & Dual Audio", 
            font=("Arial", 9, "italic"), 
            fg="#7f8c8d", 
            bg="#f0f0f0"
        )
        footer_label.pack()
        
        version_label = tk.Label(
            footer_frame, 
            text="Version 2.2 - Enhanced Edition with Dual Audio", 
            font=("Arial", 8), 
            fg="#95a5a6", 
            bg="#f0f0f0"
        )
        version_label.pack(pady=(2, 0))
    
    # Color picker method
    def choose_color(self, color_var, color_button, callback):
        """Open color picker dialog."""
        from tkinter import colorchooser
        
        color = colorchooser.askcolor(
            initialcolor=color_var.get(),
            title="Choose Text Color"
        )
        
        if color[1]:  # If user didn't cancel
            color_var.set(color[1])
            color_button.config(bg=color[1])
            
            # Update color label
            if hasattr(self, 'text_color_label') and color_var == self.text_color:
                self.text_color_label.config(text=color[1])
            elif hasattr(self, 'blur_text_color_label') and color_var == self.blur_text_color:
                self.blur_text_color_label.config(text=color[1])
            elif hasattr(self, 'narasi_text_color_label') and color_var == self.narasi_text_color:
                self.narasi_text_color_label.config(text=color[1])
            
            # Call the callback function
            if callback:
                callback()
    
    # Event handlers and utility methods
    def on_audio_mode_change(self):
        """Handle audio mode change (original only vs background only)."""
        if self.audio_enabled.get():
            # Background music mode
            self.select_audio_folder_btn.config(state=tk.NORMAL)
            self.background_volume_scale.config(state=tk.NORMAL)
            self.audio_folder_label.config(fg="#2c3e50")
            
            # Disable dual audio when background only is selected
            self.dual_audio_enabled.set(False)
            self.original_volume_scale.config(state=tk.DISABLED)
        else:
            # Original audio only mode
            self.select_audio_folder_btn.config(state=tk.DISABLED)
            self.background_volume_scale.config(state=tk.DISABLED)
            self.audio_folder_label.config(fg="#7f8c8d")
            
            # Disable dual audio when original only is selected
            self.dual_audio_enabled.set(False)
            self.original_volume_scale.config(state=tk.DISABLED)
    
    def on_dual_audio_change(self):
        """Handle dual audio mode change."""
        if self.dual_audio_enabled.get():
            # Dual audio mode enabled
            self.audio_enabled.set(False)  # Disable background only mode
            self.select_audio_folder_btn.config(state=tk.NORMAL)
            self.original_volume_scale.config(state=tk.NORMAL)
            self.background_volume_scale.config(state=tk.NORMAL)
            self.audio_folder_label.config(fg="#2c3e50")
        else:
            # Dual audio mode disabled
            self.original_volume_scale.config(state=tk.DISABLED)
            # Keep background volume enabled if background mode is selected
            if not self.audio_enabled.get():
                self.background_volume_scale.config(state=tk.DISABLED)
                self.select_audio_folder_btn.config(state=tk.DISABLED)
                self.audio_folder_label.config(fg="#7f8c8d")
    
    def select_audio_folder(self):
        """Select audio folder."""
        path = filedialog.askdirectory(title="Select Audio Folder")
        if path:
            self.audio_folder_path = path
            self.audio_folder_label.config(text=f"Audio Folder: {os.path.basename(self.audio_folder_path)}")
            
            audio_files = get_audio_files(self.audio_folder_path)
            self.audio_count_label.config(text=f"Found {len(audio_files)} audio files")
    
    def on_font_change(self, event=None):
        """Handle font change for green screen mode."""
        self.update_preview()
    
    def on_blur_font_change(self, event=None):
        """Handle font change for blur mode."""
        self.on_blur_text_change()
    
    def on_narasi_font_change(self, event=None):
        """Handle font change for narasi mode."""
        self.update_preview()
    
    def on_blur_text_change(self, value=None):
        """Handle blur text settings change."""
        if self.processing_mode.get() == "blur":
            if self.is_single_file_mode and self.selected_files:
                self.update_blur_preview_from_file()
            else:
                self.update_blur_preview()
    
    def on_narasi_position_change(self, value):
        """Handle position change for narasi mode."""
        self.update_preview()
    
    def select_folder(self):
        """Select video folder."""
        path = filedialog.askdirectory(title="Select Video/GIF Folder")
        if path:
            self.folder_path = path
            self.selected_files = []  # Clear single file selection
            self.is_single_file_mode = False
            
            self.folder_label.config(text=f"Folder: {os.path.basename(self.folder_path)}")
            
            # Count both videos and GIFs properly
            videos = get_video_files(self.folder_path)
            gifs = get_gif_files(self.folder_path)
            total_files = len(videos) + len(gifs)
            
            print(f"📊 Folder scan results:")
            print(f"   Videos: {len(videos)}")
            print(f"   GIFs: {len(gifs)}")
            print(f"   Total: {total_files}")
            
            self.video_count_label.config(text=f"Found {len(videos)} videos and {len(gifs)} GIFs ({total_files} total)")
            
            if self.processing_mode.get() == "blur":
                self.update_blur_preview()
    
    def select_files(self):
        """Select individual video/GIF files."""
        files = filedialog.askopenfilenames(
            title="Select Video/GIF Files",
            filetypes=[
                ("All Supported", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.gif"),
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("GIF Files", "*.gif"),
                ("All Files", "*.*")
            ]
        )
        
        if files:
            self.selected_files = list(files)
            self.folder_path = ""  # Clear folder selection
            self.is_single_file_mode = True
            
            # Count videos and GIFs
            videos = [f for f in self.selected_files if not is_gif_file(f)]
            gifs = [f for f in self.selected_files if is_gif_file(f)]
            
            print(f"📊 File selection results:")
            print(f"   Videos: {len(videos)}")
            print(f"   GIFs: {len(gifs)}")
            print(f"   Total: {len(self.selected_files)}")
            print(f"   Selected files: {[os.path.basename(f) for f in self.selected_files]}")
            
            file_names = [os.path.basename(f) for f in self.selected_files]
            if len(file_names) > 3:
                display_names = file_names[:3] + [f"... and {len(file_names)-3} more"]
            else:
                display_names = file_names
            
            self.folder_label.config(text=f"Files: {', '.join(display_names)}")
            self.video_count_label.config(text=f"Selected {len(videos)} videos and {len(gifs)} GIFs ({len(self.selected_files)} total)")
            
            if self.processing_mode.get() == "blur" and self.selected_files:
                self.update_blur_preview_from_file()
    
    def on_position_change(self, value):
        """Handle position change for green screen mode."""
        self.update_preview()
    
    def on_crop_change(self, value):
        """Handle crop change for blur mode."""
        if self.is_single_file_mode and self.selected_files:
            self.update_blur_preview_from_file()
        else:
            self.update_blur_preview()
    
    def update_preview(self):
        """Update preview."""
        if self.processing_mode.get() == "greenscreen":
            text_settings = self.get_text_settings()
            self.template_section.update_preview(text_settings)
        elif self.processing_mode.get() == "narasi":
            text_settings = self.get_narasi_text_settings()
            self.template_section.update_preview(text_settings)
    
    def update_blur_preview(self):
        """Update blur preview from folder."""
        if self.processing_mode.get() != "blur" or not self.folder_path:
            return
        
        try:
            # Get first video or GIF file
            videos = get_video_files(self.folder_path)
            gifs = get_gif_files(self.folder_path)
            
            if not videos and not gifs:
                self.blur_preview_label.config(text="No videos or GIFs found")
                return
            
            # Use first available file
            if videos:
                first_file = os.path.join(self.folder_path, videos[0])
                self._update_blur_preview_from_path(first_file)
            else:
                first_file = os.path.join(self.folder_path, gifs[0])
                self._update_blur_preview_from_path(first_file)
                
        except Exception as e:
            print(f"Blur preview error: {e}")
            self.blur_preview_label.config(text=f"Error: {str(e)}")
    
    def update_blur_preview_from_file(self):
        """Update blur preview from selected files."""
        if self.processing_mode.get() != "blur" or not self.selected_files:
            return
        
        try:
            # Use first selected file
            first_file = self.selected_files[0]
            self._update_blur_preview_from_path(first_file)
                
        except Exception as e:
            print(f"Blur preview error: {e}")
            self.blur_preview_label.config(text=f"Error: {str(e)}")
    
    def _update_blur_preview_from_path(self, file_path):
        """Update blur preview from specific file path with text overlay."""
        try:
            if is_gif_file(file_path):
                # Use GIF
                from .gif_processing import extract_gif_frames
                frames, _ = extract_gif_frames(file_path)
                if frames:
                    frame = frames[0]
                    ret = True
                else:
                    ret = False
            else:
                # Use video
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
            
            if not ret:
                self.blur_preview_label.config(text="Error loading file")
                return
            
            preview_width = 160
            preview_height = 285
            
            # Create blurred background
            blurred_bg = cv2.GaussianBlur(frame, (51, 51), 0)
            blurred_bg = cv2.resize(blurred_bg, (preview_width, preview_height))
            
            # Crop video
            h, w = frame.shape[:2]
            crop_top_px = int(h * self.crop_top.get() / 100)
            crop_bottom_px = int(h * self.crop_bottom.get() / 100)
            
            cropped_frame = frame[crop_top_px:h-crop_bottom_px, :]
            
            # Calculate video size
            cropped_h, cropped_w = cropped_frame.shape[:2]
            target_ratio = 9/16
            current_ratio = cropped_w / cropped_h
            
            if current_ratio > target_ratio:
                new_height = int(preview_height * 0.6)
                new_width = int(new_height * current_ratio)
            else:
                new_width = int(preview_width * 0.7)
                new_height = int(new_width / current_ratio)
            
            resized_video = cv2.resize(cropped_frame, (new_width, new_height))
            
            # Position video
            x_percent = self.video_x_position.get() / 100
            y_percent = self.video_y_position.get() / 100
            
            max_x = preview_width - new_width
            max_y = preview_height - new_height
            
            x_offset = int(x_percent * max_x) if max_x > 0 else 0
            y_offset = int(y_percent * max_y) if max_y > 0 else 0
            
            x_offset = max(0, min(x_offset, max_x))
            y_offset = max(0, min(y_offset, max_y))
            
            # Overlay video
            result = blurred_bg.copy()
            result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_video
            
            # Convert to PIL for text overlay
            pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # Add basic labels
            try:
                font = ImageFont.truetype("arial.ttf", 8)
            except:
                font = ImageFont.load_default()
            
            draw.text((3, 3), "BLUR BG", fill="white", font=font)
            draw.text((x_offset + 3, y_offset + 3), "VIDEO", fill="white", font=font)
            
            # Add text overlay preview if enabled
            if self.blur_text_enabled.get():
                try:
                    preview_font_size = max(8, int(self.blur_text_size.get() * 0.15))
                    
                    try:
                        text_font = ImageFont.truetype("arial.ttf", preview_font_size)
                    except:
                        text_font = ImageFont.load_default()
                    
                    sample_text = f"Sample Text ({self.blur_selected_font.get()})"
                    max_width = pil_img.width - 10
                    lines = smart_text_wrap(sample_text, draw, text_font, max_width, emoji_size=15)
                    
                    line_height = preview_font_size + 3
                    total_height = len(lines) * line_height
                    base_y = int((self.blur_text_y_position.get() / 100) * (pil_img.height - total_height - 10))
                    base_y = max(5, min(base_y, pil_img.height - total_height - 5))
                    
                    rendered_lines = render_text_with_emoji_multiline(
                        draw, lines, text_font, pil_img.width, pil_img.height, 
                        base_y, emoji_size=15, line_spacing=3
                    )
                    
                    # Get text color
                    text_color = self.blur_text_color.get()
                    
                    for line_data in rendered_lines:
                        for item_type, item, x_offset_text in line_data['items']:
                            if item_type == 'emoji':
                                emoji_y = line_data['y'] + (preview_font_size - line_data['emoji_size']) // 2
                                pil_img.paste(item, (line_data['x_start'] + x_offset_text, emoji_y), item)
                            elif item_type == 'text':
                                draw.text((line_data['x_start'] + x_offset_text, line_data['y']), 
                                         item, fill=text_color, font=text_font)
                except Exception as text_error:
                    print(f"Text preview error: {text_error}")
            
            photo = ImageTk.PhotoImage(pil_img)
            self.blur_preview_label.config(image=photo, text="", width=preview_width, height=preview_height)
            self.blur_preview_label.image = photo
            
        except Exception as e:
            print(f"Blur preview error: {e}")
            self.blur_preview_label.config(text=f"Error: {str(e)}")
    
    def update_progress(self, current, total, file_name):
        """Update progress."""
        progress = (current / total) * 100
        self.progress_bar['value'] = progress
        self.status_label.config(text=f"Processing: {file_name} ({current}/{total})")
        self.root.update_idletasks()
    
    def set_process_callback(self, callback):
        """Set process callback."""
        self.process_btn.config(command=callback)
    
    # Utility methods
    def configure_responsive_grid(self, parent, columns=2):
        """Configure responsive grid layout."""
        for i in range(columns):
            parent.grid_columnconfigure(i, weight=1, uniform="column")
    
    # Getter methods for settings
    def get_text_settings(self):
        """Get text settings for green screen mode."""
        return {
            'enabled': self.text_enabled.get(),
            'x_position': self.text_x_position.get(),
            'y_position': self.text_y_position.get(),
            'size': self.text_size.get(),
            'font': self.selected_font.get(),
            'color': self.text_color.get()  # NEW: Include color
        }
    
    def get_narasi_text_settings(self):
        """Get text settings for narasi mode."""
        return {
            'enabled': self.narasi_text_enabled.get(),
            'x_position': self.narasi_text_x_position.get(),
            'y_position': self.narasi_text_y_position.get(),
            'size': self.narasi_text_size.get(),
            'font': self.narasi_selected_font.get(),
            'color': self.narasi_text_color.get()  # NEW: Include color
        }
    
    def get_blur_text_settings(self):
        """Get text settings for blur mode."""
        return {
            'enabled': self.blur_text_enabled.get(),
            'x_position': self.blur_text_x_position.get(),
            'y_position': self.blur_text_y_position.get(),
            'size': self.blur_text_size.get(),
            'font': self.blur_selected_font.get(),
            'color': self.blur_text_color.get()  # NEW: Include color
        }
    
    def get_blur_settings(self):
        """Get blur settings."""
        return {
            'crop_top': self.crop_top.get(),
            'crop_bottom': self.crop_bottom.get(),
            'video_x_position': self.video_x_position.get(),
            'video_y_position': self.video_y_position.get()
        }
    
    def get_audio_settings(self):
        """Get enhanced audio settings with dual audio support."""
        return {
            'enabled': self.audio_enabled.get(),
            'dual_audio_enabled': self.dual_audio_enabled.get(),  # NEW
            'folder_path': self.audio_folder_path,
            'original_volume': self.original_audio_volume.get(),  # NEW
            'background_volume': self.background_audio_volume.get()  # RENAMED from 'volume'
        }
    
    def get_gpu_settings(self):
        """Get GPU settings."""
        return {
            'enabled': self.gpu_enabled.get() and gpu_config.GPU_AVAILABLE,
            'encoder': self.selected_encoder.get().replace(" (CPU)", ""),
            'decoder': self.selected_decoder.get() if self.selected_decoder.get() != "CPU" else None,
            'config': gpu_config.get_config_summary()
        }
    
    def get_processing_mode(self):
        """Get processing mode."""
        return self.processing_mode.get()
    
    def get_narasi_audio_path(self):
        """Get narasi audio path."""
        return self.narasi_section.get_narasi_audio_path()
    
    def get_files_to_process(self):
        """Get list of files to process (either from folder or selected files)."""
        if self.is_single_file_mode and self.selected_files:
            # Return selected files with their full paths
            return [(os.path.dirname(f), os.path.basename(f)) for f in self.selected_files]
        elif self.folder_path:
            # Return files from folder
            videos = get_video_files(self.folder_path)
            gifs = get_gif_files(self.folder_path)
            all_files = videos + gifs
            return [(self.folder_path, f) for f in all_files]
        else:
            return []