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

class Section4Scene(TeachingScene):
    def construct(self):
        # DATA FOR SECTION 4
        title = "The Activation Gate: Filtering Knowledge"
        lines = [
            "The GELU function acts as a nonlinear activation gate.",
            "Only strong matches pass through to the next stage.",
            "This filter suppresses noise and irrelevant factual data."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_THRESHOLD_OFF = "#FF0000" # Red
        COLOR_THRESHOLD_ON = "#00FF00"  # Green
        COLOR_PULSE = "#FFFF00"         # Yellow

        # === Animation for Lecture Line 1 ===
        # Description: Draw a red horizontal line #FF0000 labeled "Activation Threshold".
        self.lecture[0].set_color(YELLOW)
        
        threshold_line = Line(
            start=self.grid["C2"], 
            end=self.grid["C5"], 
            color=COLOR_THRESHOLD_OFF, 
            stroke_width=6
        )
        threshold_label = Text("Activation Threshold", font_size=20, color=COLOR_THRESHOLD_OFF)
        # Position label away from the center to avoid overlap with the pulse path (Column 3)
        # Fix for Issue 37: Moved from B2-B5 to B4-B6 and scaled down to 0.8
        self.place_in_area(threshold_label, "B4", "B6", scale_factor=0.8)
        
        self.play(Create(threshold_line), Write(threshold_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Description: Move a bright signal pulse from below toward the line.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Create signal pulse at E3
        pulse_core = Dot(color=COLOR_PULSE, radius=0.15)
        pulse_glow = Dot(color=COLOR_PULSE, radius=0.3, fill_opacity=0.3)
        pulse = VGroup(pulse_glow, pulse_core)
        # Fix for Issue 38: Scaled pulse down to 0.6 to avoid obscuring elements
        self.place_at_grid(pulse, "E3", scale_factor=0.6)
        
        self.play(FadeIn(pulse))
        # Move toward the line at C3
        self.play(pulse.animate.move_to(self.grid["C3"]), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Description: The line turns green #00FF00 as the pulse passes through.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Move pulse through the gate from C3 to A3
        # Simultaneous color change of threshold line and label
        self.play(
            pulse.animate.move_to(self.grid["A3"]),
            threshold_line.animate.set_color(COLOR_THRESHOLD_ON),
            threshold_label.animate.set_color(COLOR_THRESHOLD_ON),
            run_time=1.5
        )
        
        # Briefly show "noise" being suppressed (as mentioned in line text)
        noise_dots = VGroup(*[
            Dot(color=GRAY, radius=0.08, fill_opacity=0.4).move_to(self.grid["F2"] + np.array([i*0.8, 0, 0]))
            for i in range(4)
        ])
        self.play(FadeIn(noise_dots))
        self.play(noise_dots.animate.shift(UP * 0.4), rate_func=wiggle)
        self.play(FadeOut(noise_dots))
        
        self.wait(2)
