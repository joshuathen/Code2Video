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
        lecture_lines = [
            "Light always follows the path of least time.",
            "It refracts when passing through different media.",
            "Gravity acts like a medium with changing speed."
        ]
        self.setup_layout("Fermat's Principle and the Light Analogy", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Layers representing different media
        layer_colors = ["#1A237E", "#283593", "#303F9F", "#3949AB"] 
        layers = VGroup()
        for i in range(4):
            # Each layer spans one row height (1.0) and 6 units width
            rect = Rectangle(width=6, height=1, fill_color=layer_colors[i], fill_opacity=0.3, stroke_width=1, stroke_color=BLUE_A)
            row_char = chr(ord('A') + i)
            self.place_in_area(rect, f"{row_char}1", f"{row_char}6")
            layers.add(rect)
        
        self.play(FadeIn(layers))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Points for discrete refraction path
        # Start top-leftish in Layer A
        pts = [
            np.array([1.0, 2.7, 0]),   # Start
            np.array([1.27, 1.7, 0]),  # A/B
            np.array([1.85, 0.7, 0]),  # B/C
            np.array([2.85, -0.3, 0]), # C/D
            np.array([4.58, -1.3, 0])  # End
        ]
        
        path_segments = VGroup(*[
            Line(pts[i], pts[i+1], color=YELLOW) for i in range(len(pts)-1)
        ])
        
        self.play(Create(path_segments, run_time=3))
        self.wait(1)
        
        # Transition to many layers and smooth curve
        r_val = 1.5
        cycloid = ParametricFunction(
            lambda t: np.array([
                r_val * (t - np.sin(t)) + 1.0,
                -r_val * (1 - np.cos(t)) + 2.7,
                0
            ]),
            t_range=[0, np.pi],
            color=GOLD
        )
        
        many_layers = VGroup(*[
            Line(start=[0.5, 2.7 - k*0.1, 0], end=[5.5, 2.7 - k*0.1, 0], 
                 stroke_width=0.5, stroke_opacity=0.15, color=BLUE_B)
            for k in range(40)
        ])
        
        self.play(
            FadeOut(layers),
            FadeOut(path_segments),
            FadeIn(many_layers),
            Create(cycloid)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Label curve 'Path of Least Time' in gold
        label = Text("Path of Least Time", font_size=20, color="#FFD700")
        # Fix for Issue 35: Use place_in_area for multi-word label
        self.place_in_area(label, 'F4', 'F6', scale_factor=0.8)
        
        # Marble sliding (Asset Integration - Issue 26)
        marble = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg").scale(0.3)
        # Use persistent mobject + ValueTracker for movement
        t_tracker = ValueTracker(0)
        # Apply initial position before adding updater
        marble.move_to(cycloid.point_from_proportion(0))
        marble.add_updater(lambda m: m.move_to(cycloid.point_from_proportion(t_tracker.get_value())))
        
        self.add(marble)
        self.play(Write(label))
        self.play(t_tracker.animate.set_value(1), run_time=3, rate_func=slow_into)
        self.wait(2)
        
        # Final color reset
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
