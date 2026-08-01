from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section6Scene(TeachingScene):
    def construct(self):
        # Teaching script and title
        lecture_lines = [
            "Fourier Transforms power modern digital technology.",
            "They enable efficient data compression for JPEGs.",
            "This tool is the foundation of digital communication."
        ]
        self.setup_layout("Summary and Digital Impact", lecture_lines)
        
        # Define colors
        CYAN = "#00FFFF"
        YELLOW = "#FFFF00"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Cyan icons (#00FFFF) for MP3, JPEG, and Wi-Fi float around a prism.
        self.lecture[0].set_color(CYAN)
        
        prism = Triangle(color=WHITE, fill_opacity=0.1)
        # Fix Issue 34: self.place_at_grid(prism, 'B3', scale_factor=1.0)
        self.place_at_grid(prism, 'B3', scale_factor=1.0)
        
        icon_mp3 = VGroup(
            RoundedRectangle(height=0.6, width=1.1, color=CYAN), 
            Text("MP3", font_size=18, color=CYAN)
        )
        icon_jpeg = VGroup(
            RoundedRectangle(height=0.6, width=1.1, color=CYAN), 
            Text("JPEG", font_size=18, color=CYAN)
        )
        icon_wifi = VGroup(
            RoundedRectangle(height=0.6, width=1.1, color=CYAN), 
            Text("Wi-Fi", font_size=18, color=CYAN)
        )
        
        self.place_at_grid(icon_mp3, "A2")
        self.place_at_grid(icon_jpeg, "A5")
        # Fix Issue 33: self.place_at_grid(icon_wifi, 'B2')
        self.place_at_grid(icon_wifi, "B2")
        
        # Persistent mobjects for floating effect
        time_tracker = ValueTracker(0)
        self.add(time_tracker)
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        
        # Save base positions for oscillating update
        pos_mp3 = icon_mp3.get_center().copy()
        pos_jpeg = icon_jpeg.get_center().copy()
        pos_wifi = icon_wifi.get_center().copy()
        
        icon_mp3.add_updater(lambda m: m.move_to(pos_mp3 + UP * 0.15 * np.sin(time_tracker.get_value() * 2)))
        icon_jpeg.add_updater(lambda m: m.move_to(pos_jpeg + DOWN * 0.15 * np.cos(time_tracker.get_value() * 1.5)))
        icon_wifi.add_updater(lambda m: m.move_to(pos_wifi + RIGHT * 0.15 * np.sin(time_tracker.get_value() * 2.2)))
        
        self.play(Create(prism), FadeIn(icon_mp3), FadeIn(icon_jpeg), FadeIn(icon_wifi))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # A large stack of data blocks shrinks into a single yellow byte icon (#FFFF00).
        self.lecture[1].set_color(YELLOW)
        
        # Representing raw data blocks
        blocks = VGroup(*[
            Square(side_length=0.3, fill_opacity=0.6, fill_color=GREY_A, stroke_width=1) 
            for _ in range(12)
        ])
        blocks.arrange_in_grid(rows=3, cols=4, buff=0.1)
        # Fix Issue 33: self.place_in_area(blocks, 'D2', 'E3')
        self.place_in_area(blocks, "D2", "E3")
        
        byte_icon = VGroup(
            RoundedRectangle(height=0.7, width=0.7, color=YELLOW, fill_opacity=0.2), 
            Text("01", font_size=20, color=YELLOW)
        )
        # Fix Issue 35: self.place_at_grid(byte_icon, 'F2', scale_factor=1.0)
        self.place_at_grid(byte_icon, "F2", scale_factor=1.0)
        
        self.play(Create(blocks))
        self.wait(1)
        # Compression animation
        self.play(ReplacementTransform(blocks, byte_icon))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # The title 'Fourier Transform: The Recipe of Reality' fades in for the finale.
        self.lecture[2].set_color(WHITE_COLOR)
        
        final_title = Text(
            "Fourier Transform:\nThe Recipe of Reality", 
            font_size=36, 
            color=WHITE_COLOR, 
            line_spacing=1.2
        )
        # Fix Issue 35: self.place_in_area(final_title, 'B4', 'E6')
        self.place_in_area(final_title, "B4", "E6")
        
        # Final sequence: Fade out distractions and show grand title
        self.play(
            FadeOut(prism), 
            FadeOut(icon_mp3), 
            FadeOut(icon_jpeg), 
            FadeOut(icon_wifi),
            FadeOut(byte_icon),
            FadeIn(final_title)
        )
        self.wait(5)
