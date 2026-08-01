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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "The Daily Secret & Rolling IDs"
        lines = [
            "Every day, Pip’s phone generates a Secret Day Key.",
            "This key derives a unique code every fifteen minutes.",
            "Frequent rotation prevents tracking by any malicious observer."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_SK = "#FFA500"  # Secret Key Orange
        COLOR_RPI = "#00FFFF" # RPI Cyan
        COLOR_HIGHLIGHT = YELLOW

        # === Animation for Lecture Line 1 ===
        # Pip's phone icon containing a 'Secret Day Key'
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))
        
        phone_body = RoundedRectangle(corner_radius=0.2, height=2.5, width=1.5, color=WHITE)
        self.place_in_area(phone_body, "B2", "D3")
        
        sk_label = Text("SK_t", font_size=24, color=COLOR_SK)
        # Issue 38 Fix: Centering sk_label inside the phone body area
        self.place_in_area(sk_label, "B2", "D3", scale_factor=0.8)
        
        sk_box = SurroundingRectangle(sk_label, color=COLOR_SK, buff=0.1)
        phone_group = VGroup(phone_body, sk_label, sk_box)
        
        self.play(Create(phone_body))
        self.play(FadeIn(sk_label), Create(sk_box))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A code 'A12B' pulses from the phone with a '10:00 AM' timestamp.
        # Then changes to 'C99X' with a '10:15 AM' timestamp.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # First RPI
        # Issue 39 Fix: Increased scale_factor to 1.2
        rpi_1 = Text("A12B", font_size=32, color=COLOR_RPI)
        self.place_at_grid(rpi_1, "B5", scale_factor=1.2)
        
        time_1 = Text("10:00 AM", font_size=20, color=GRAY)
        self.place_at_grid(time_1, "C5")
        
        arrow_1 = Arrow(start=self.grid["C3"], end=self.grid["C5"], color=COLOR_RPI)
        
        self.play(GrowArrow(arrow_1))
        self.play(FadeIn(rpi_1), FadeIn(time_1))
        self.play(rpi_1.animate.scale(1.2), run_time=0.5, rate_func=there_and_back)
        self.wait(1)

        # Transition to second RPI
        # Issue 39 Fix: Increased scale_factor to 1.2
        rpi_2 = Text("C99X", font_size=32, color=COLOR_RPI)
        self.place_at_grid(rpi_2, "B5", scale_factor=1.2)
        
        time_2 = Text("10:15 AM", font_size=20, color=GRAY)
        self.place_at_grid(time_2, "C5")
        
        self.play(
            Transform(rpi_1, rpi_2),
            Transform(time_1, time_2)
        )
        self.play(rpi_1.animate.scale(1.2), run_time=0.5, rate_func=there_and_back)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Frequent rotation prevents tracking by any malicious observer.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Add a "Stranger" icon/text to emphasize anonymity
        stranger_label = Text("Unknown", font_size=24, color=RED)
        self.place_at_grid(stranger_label, "E5")
        
        eye_icon = VGroup(
            Ellipse(width=0.8, height=0.4, color=WHITE),
            Dot(color=WHITE)
        )
        # Issue 40 Fix: Increased scale_factor to 1.2
        self.place_at_grid(eye_icon, "E4", scale_factor=1.2)
        
        self.play(FadeIn(eye_icon), Write(stranger_label))
        
        # Flash RPI to show it looks disconnected to the observer
        self.play(Indicate(rpi_1, color=COLOR_RPI))
        self.wait(2)
        
        # Final cleanup highlight
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
