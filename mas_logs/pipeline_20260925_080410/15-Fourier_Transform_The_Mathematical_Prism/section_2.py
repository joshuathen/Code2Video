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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Periodic signals repeat over time.", "Represent them with sine waves.", "Rotation creates this wave pattern."]
        self.setup_layout("Prerequisite Review: Periodic Motion", lecture_lines)
        
        # --- Pre-calculate elements ---
        # Wave
        axes = Axes(x_range=[0, 4*PI, 1], y_range=[-2, 2, 1], x_length=4, y_length=2.5, axis_config={"include_tip": False})
        sine_wave = axes.plot(lambda t: np.sin(t), color="#00FF00")
        
        # Assets
        pendulum = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg")
        gear = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gear.svg")
        
        # Labels
        amplitude_label = Text("Amplitude", font_size=18, color=WHITE)
        period_label = Text("Period", font_size=18, color=WHITE)
        
        # Positioning placeholders
        self.place_at_grid(axes, 'D4', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        self.place_at_grid(pendulum, 'B4', scale_factor=0.5)
        self.play(Create(axes), Create(sine_wave), FadeIn(pendulum))
        # Pendulum simple swing
        self.play(Rotate(pendulum, angle=PI/4, about_point=self.grid['B4'] + UP*0.5), run_time=1)
        self.play(Rotate(pendulum, angle=-PI/2, about_point=self.grid['B4'] + UP*0.5), run_time=1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        # Amplitude label
        self.place_at_grid(amplitude_label, 'C4', scale_factor=0.8)
        self.play(Write(amplitude_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        # Period label
        self.place_at_grid(period_label, 'E5', scale_factor=0.8)
        self.place_at_grid(gear, 'F5', scale_factor=0.5)
        self.play(Write(period_label), FadeIn(gear))
        self.play(Rotate(gear, angle=2*PI, run_time=2))
        self.wait(1)
