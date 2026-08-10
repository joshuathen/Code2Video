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
        lines = ["Cheetahs run 100 meters in 5 seconds.", "Average speed is distance divided by time.", "100 meters divided by 5 seconds is 20 m/s."]
        self.setup_layout("The Problem of Average Speed", lines)
        
        # Define the curve (position vs time)
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 120, 20], axis_config={"include_numbers": False}).scale(0.5)
        curve = axes.plot(lambda t: 4 * t**2, x_range=[0, 5], color=BLUE)
        
        # Add cheetah asset
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_in_area(axes, 'B1', 'D4', scale_factor=0.7) # Addressed Issue 22
        self.place_at_grid(cheetah, 'E5', scale_factor=0.5) # Asset integration
        
        group = VGroup(axes, curve, cheetah)
        self.play(Create(axes), Create(curve), FadeIn(cheetah), run_time=2)
        self.wait(2)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        p1 = axes.c2p(0, 0)
        p2 = axes.c2p(5, 100)
        secant = Line(p1, p2, color=WHITE)
        
        self.play(Create(secant), run_time=2)
        self.wait(2)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.play(secant.animate.set_color("#FF4500"), run_time=2)
        
        # Add calculation text
        calc = MathTex(r"v_{avg} = \frac{100m}{5s} = 20 m/s", font_size=32)
        self.place_at_grid(calc, 'E6', scale_factor=0.8) # Addressed Issue 21, 23, 36
        self.play(Write(calc), run_time=2)
        
        self.wait(8)
