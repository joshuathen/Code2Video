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
        # Initial Setup
        title = "Prerequisite: Symmetry and Periodicity"
        lines = [
            "Square waves are odd functions, symmetric around the origin.",
            "This symmetry means we only need pure sine waves.",
            "No cosines are required, simplifying our mathematical toolkit."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        SINE_COLOR = "#00FF00"
        SQUARE_COLOR = "#FF0000"
        AXES_COLOR = BLUE_D
        LABEL_COLOR = YELLOW_A
        
        # === Animation for Lecture Line 1 ===
        # "Square waves are odd functions, symmetric around the origin."
        self.lecture[0].set_color(SINE_COLOR)
        
        axes = Axes(
            x_range=[-PI, PI, PI/2],
            y_range=[-1.5, 1.5, 1],
            x_length=4,
            y_length=3,
            axis_config={"color": AXES_COLOR, "include_tip": True}
        )
        self.place_in_area(axes, "B2", "E5")
        
        origin_dot = Dot(axes.c2p(0, 0, 0), color=WHITE, radius=0.08)
        sine_wave = axes.plot(lambda x: np.sin(x), color=SINE_COLOR)
        
        self.play(Create(axes), FadeIn(origin_dot))
        self.play(Create(sine_wave))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "This symmetry means we only need pure sine waves."
        self.lecture[1].set_color(SINE_COLOR)
        
        # Ghost sine to show the starting position during rotation
        ghost_sine = sine_wave.copy().set_stroke(opacity=0.3)
        self.add(ghost_sine)
        
        # Rotate 180 degrees to demonstrate odd symmetry f(-x) = -f(x)
        self.play(
            Rotate(sine_wave, angle=PI, about_point=axes.c2p(0, 0, 0)),
            run_time=2
        )
        
        # Position fixed per Issue 30: Move odd_math to F3
        odd_math = MathTex("f(-x) = -f(x)", color=LABEL_COLOR, font_size=36)
        self.place_at_grid(odd_math, "F3", scale_factor=0.8)
        self.play(Write(odd_math))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "No cosines are required, simplifying our mathematical toolkit."
        self.lecture[2].set_color(SQUARE_COLOR)
        
        # Transformation to Square Wave
        # Square wave: 4/pi * sum(sin(n*x)/n) for odd n
        # For this demonstration, the simple signum of sine is enough
        square_wave = axes.plot(
            lambda x: 1 if np.sin(x) > 0 else -1 if np.sin(x) < 0 else 0,
            color=SQUARE_COLOR,
            use_smoothing=False
        )
        
        # Position fixed per Issue 31: Move sine_only_text to F4
        sine_only_text = Text("Sine-Only Series", color=SQUARE_COLOR, font_size=24)
        self.place_at_grid(sine_only_text, "F4", scale_factor=0.8)
        
        # Fade out old text and formula while bringing in the square wave and new text
        self.play(
            FadeOut(sine_wave),
            FadeOut(ghost_sine),
            FadeOut(odd_math),
            FadeIn(sine_only_text),
            Create(square_wave)
        )
        self.wait(1)
        
        # Final rotation to show square wave also has the same property
        self.play(
            Rotate(square_wave, angle=PI, about_point=axes.c2p(0, 0, 0)),
            run_time=2
        )
        self.wait(2)
