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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initialization
        self.setup_layout("Introduction: The Secret Recipe of Sounds", [
            "Any complex periodic signal contains hidden pure waves.",
            "Melody-Bot uses tuning forks to recreate a messy buzz.",
            "Combining pure notes yields the original complex sound."
        ])
        
        # Colors
        HIGHLIGHT_COLOR = YELLOW
        COMPLEX_COLOR = "#FF5555"
        PURE_COLOR = "#55FF55"
        FORK_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Create complex buzzing wave function
        # A messy periodic wave formed by a sum of sines
        complex_wave = FunctionGraph(
            lambda x: 0.5 * (np.sin(2 * PI * x) + 0.4 * np.sin(4 * PI * x + PI/4) + 0.2 * np.sin(8 * PI * x)),
            x_range=[0, 4],
            color=COMPLEX_COLOR
        )
        # Fix for Issue 23: Move complex_wave from A1-C6 to A4-C6
        self.place_in_area(complex_wave, "A4", "C6", scale_factor=0.8)
        
        self.play(Create(complex_wave))
        
        # Animate oscillation by shifting
        self.play(complex_wave.animate.shift(LEFT * 0.2), run_time=0.5, rate_func=linear)
        self.play(complex_wave.animate.shift(RIGHT * 0.4), run_time=1.0, rate_func=linear)
        self.play(complex_wave.animate.shift(LEFT * 0.2), run_time=0.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Character 'Melody-Bot'
        # Fix for Issue 19: Use asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        bot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_in_area(bot, "D1", "E2", scale_factor=0.6)
        
        # Tuning forks: handle + arc + prongs
        def get_tuning_fork():
            handle = Line(ORIGIN, UP*0.4, color=FORK_COLOR)
            u_shape = Arc(radius=0.2, start_angle=PI, angle=PI, color=FORK_COLOR).shift(UP*0.4)
            prongs = VGroup(
                Line(u_shape.get_start(), u_shape.get_start() + UP*0.4, color=FORK_COLOR),
                Line(u_shape.get_end(), u_shape.get_end() + UP*0.4, color=FORK_COLOR)
            )
            return VGroup(handle, u_shape, prongs)

        forks = VGroup(get_tuning_fork(), get_tuning_fork(), get_tuning_fork()).arrange(RIGHT, buff=0.4)
        # Fix for Issue 24: Move forks from E4 to D4
        self.place_at_grid(forks, "D4", scale_factor=0.7)
        
        self.play(FadeIn(bot), FadeIn(forks))
        
        # Vibrate forks animation
        self.play(forks.animate.scale(1.1), rate_func=wiggle, run_time=1)
        self.play(forks.animate.scale(1/1.1), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Three pure sine waves (components of the complex signal)
        s1 = FunctionGraph(lambda x: 0.5 * np.sin(2 * PI * x), x_range=[0, 4], color=PURE_COLOR)
        s2 = FunctionGraph(lambda x: 0.2 * np.sin(4 * PI * x), x_range=[0, 4], color=PURE_COLOR)
        s3 = FunctionGraph(lambda x: 0.1 * np.sin(8 * PI * x), x_range=[0, 4], color=PURE_COLOR)
        
        pure_waves = VGroup(s1, s2, s3).arrange(DOWN, buff=0.3)
        # Fix for Issue 25: Move pure_waves from E3-F6 to E4-F6
        self.place_in_area(pure_waves, "E4", "F6", scale_factor=0.6)
        
        self.play(Create(pure_waves))
        self.wait(1)
        
        # Merge effect: pure waves move to the complex wave and fade out while the complex wave thickens
        self.play(
            pure_waves.animate.move_to(complex_wave.get_center()).set_opacity(0),
            complex_wave.animate.set_stroke(width=6),
            run_time=2
        )
        
        self.wait(2)
