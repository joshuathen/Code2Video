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
        # Setup the scene
        self.setup_layout(
            "Summary: The Universal Decoder", 
            [
                'From smoothies to signals, everything has hidden ingredients.', 
                "The Fourier Transform decodes the universe's complex patterns.", 
                'It powers modern technology from WiFi to medical imaging.'
            ]
        )
        
        # Define Colors
        COLOR_TIME = "#FFFF00"      # Yellow
        COLOR_FREQ = "#00FFFF"      # Cyan
        COLOR_BRIDGE = "#FFFFFF"    # White glow
        COLOR_WIFI = "#FF8C00"      # Orange
        COLOR_JPEG = "#FF00FF"      # Magenta
        COLOR_MRI = "#00FF00"       # Green

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_TIME)
        
        # Create Time and Frequency containers
        time_circle = Circle(radius=1.0, color=COLOR_TIME, stroke_width=4)
        time_label = Text("Time", font_size=20, color=WHITE).next_to(time_circle, DOWN, buff=0.2)
        time_group = VGroup(time_circle, time_label)
        # Updated layout to prevent crowding near lecture notes
        self.place_in_area(time_group, "B2", "D3", scale_factor=0.8)
        
        freq_circle = Circle(radius=1.0, color=COLOR_FREQ, stroke_width=4)
        freq_label = Text("Frequency", font_size=20, color=WHITE).next_to(freq_circle, DOWN, buff=0.2)
        freq_group = VGroup(freq_circle, freq_label)
        # Updated layout to prevent crowding near the right edge
        self.place_in_area(freq_group, "B4", "D5", scale_factor=0.8)
        
        self.play(
            Create(time_circle),
            Write(time_label),
            Create(freq_circle),
            Write(freq_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_FREQ)
        
        # Create Bridge
        # The bridge connects the perimeters of the circles
        bridge_start = time_circle.get_right()
        bridge_end = freq_circle.get_left()
        
        # Create a glowing bridge using multiple lines
        bridge_main = Line(bridge_start, bridge_end, color=COLOR_BRIDGE, stroke_width=6)
        bridge_glow = Line(bridge_start, bridge_end, color=COLOR_BRIDGE, stroke_width=12, stroke_opacity=0.3)
        bridge = VGroup(bridge_glow, bridge_main)
        
        self.play(Create(bridge), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_MRI)
        
        # Icons construction
        # WiFi Icon
        wifi_dot = Dot(radius=0.05, color=COLOR_WIFI)
        wifi_arc1 = Arc(radius=0.15, angle=PI/2, start_angle=PI/4, color=COLOR_WIFI)
        wifi_arc2 = Arc(radius=0.25, angle=PI/2, start_angle=PI/4, color=COLOR_WIFI)
        wifi_icon = VGroup(wifi_dot, wifi_arc1, wifi_arc2).scale(0.8)
        
        # JPEG Icon
        jpeg_rect = RoundedRectangle(corner_radius=0.05, height=0.4, width=0.6, color=COLOR_JPEG)
        jpeg_text = Text("JPEG", font_size=10, color=WHITE).move_to(jpeg_rect)
        jpeg_icon = VGroup(jpeg_rect, jpeg_text)
        
        # MRI Icon
        mri_circle = Circle(radius=0.2, color=COLOR_MRI)
        mri_cross = Cross(mri_circle, stroke_width=2, scale_factor=0.5).set_color(COLOR_MRI)
        mri_icon = VGroup(mri_circle, mri_cross)

        # Position icons at start of bridge
        icons = [wifi_icon, jpeg_icon, mri_icon]
        for icon in icons:
            icon.move_to(bridge_start)

        # Animation: Icons moving across the bridge one by one
        # Use successions for movement
        self.play(
            wifi_icon.animate.move_to(bridge_end).set_opacity(0),
            run_time=2, rate_func=slow_into
        )
        self.play(
            jpeg_icon.animate.move_to(bridge_end).set_opacity(0),
            run_time=2, rate_func=slow_into
        )
        self.play(
            mri_icon.animate.move_to(bridge_end).set_opacity(0),
            run_time=2, rate_func=slow_into
        )
        
        self.wait(2)
        
        # Final state: reset color to highlight the whole recap
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            run_time=0.5
        )
        self.wait(2)
