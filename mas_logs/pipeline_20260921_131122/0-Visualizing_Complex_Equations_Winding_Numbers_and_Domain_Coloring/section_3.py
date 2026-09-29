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
            "Winding number counts loops around the origin.",
            "A full loop traps the point inside.",
            "Trapped roots exist inside the closed curve."
        ]
        self.setup_layout("The Winding Number: Counting Roots", lecture_lines)
        
        # Create visual elements
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": False})
        origin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/origin.svg", color=YELLOW)
        curve = ParametricFunction(lambda t: np.array([2*np.cos(t) + 0.5*np.sin(3*t), 2*np.sin(t) + 0.5*np.cos(2*t), 0]), t_range=[0, 2*PI], color=WHITE)
        
        # Position visual elements based on VideoCritic feedback
        # Line 61 fix (using the latest suggestion from #29):
        self.place_in_area(axes, 'A3', 'D6', scale_factor=0.55)
        # Line 62 fix (from #28):
        self.place_at_grid(origin, 'D4', scale_factor=1.0)
        # Position the curve relative to axes origin after scaling
        curve.move_to(axes.get_origin())
        
        # Animations
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Create(curve))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Using origin (SVGMobject)
        self.play(Flash(origin, color=YELLOW, line_length=0.2, num_lines=12))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        dot_in = Dot(color=PURPLE).move_to(axes.c2p(0.2, 0.2))
        self.play(FadeIn(dot_in))
        self.play(Indicate(dot_in))
        self.wait(2)
