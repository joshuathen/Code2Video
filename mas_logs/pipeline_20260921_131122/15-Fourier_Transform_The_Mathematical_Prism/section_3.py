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
        lecture_lines = [
            "Euler's formula relates rotation to waves.",
            "Imagine a point tracing a perfect circle.",
            "Spinning the signal reveals hidden resonance.",
            "Matching frequencies shift the center of mass.",
            "Rotation uncovers the signal's core ingredients."
        ]
        self.setup_layout("The Core Mechanism: The Rotating Phasor", lecture_lines)
        
        # Define objects
        compass_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        metronome_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg")
        
        circle = Circle(radius=1.5, color="#FFFFFF")
        radius_line = Line(ORIGIN, RIGHT * 1.5, color="#00FF00")
        compass_icon = compass_asset.copy().set_color("#00FF00")
        
        real_axis = Arrow(LEFT * 2, RIGHT * 2, color="#00FFFF")
        imag_axis = Arrow(DOWN * 2, UP * 2, color="#FFFF00")
        
        phasor_group = VGroup(circle, radius_line, compass_icon, real_axis, imag_axis)
        self.place_in_area(phasor_group, "C3", "F6", scale_factor=0.7)
        
        # Angle tracker
        angle = ValueTracker(0)
        radius_line.add_updater(lambda m: m.set_angle(angle.get_value()).shift(circle.get_center() - m.get_start()))
        compass_icon.add_updater(lambda m: m.move_to(circle.get_center() + 1.5 * np.array([np.cos(angle.get_value()), np.sin(angle.get_value()), 0])))
        
        # Sine wave setup
        sine_wave = ParametricFunction(
            lambda t: np.array([t * 0.5, np.sin(t), 0]),
            t_range=np.array([0, 4]),
            color="#FF00FF"
        )
        self.place_in_area(sine_wave, "B5", "E6", scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(circle), Write(real_axis), Write(imag_axis), FadeIn(compass_icon))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        self.play(Create(radius_line))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        self.play(Create(sine_wave))
        self.play(angle.animate.set_value(2 * PI), run_time=3, rate_func=linear)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FFFF")
        self.play(Indicate(real_axis))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        self.play(FadeIn(metronome_asset.scale(0.5).move_to(self.grid["A4"])))
        self.play(Indicate(imag_axis))
        self.wait(1)
