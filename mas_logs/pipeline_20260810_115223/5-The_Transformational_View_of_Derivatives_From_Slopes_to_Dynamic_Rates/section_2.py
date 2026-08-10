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
        self.setup_layout("The Problem of Curvature", [
            "Curves have a changing slope.",
            "Steepness shifts at every step.",
            "How to measure local curvature?"
        ])
        
        # Setup coordinates for curve
        axes = Axes(
            x_range=[-2, 2, 1], y_range=[-1, 3, 1], 
            axis_config={"include_tip": False}
        ).scale(0.5)
        curve = axes.plot(lambda x: 0.5 * x**2 + 1, color=WHITE)
        axes_group = VGroup(axes, curve)
        
        # Grid placement (Fix for Issue 24/37)
        self.place_in_area(axes_group, 'C2', 'E5', scale_factor=0.7)
        self.add(axes_group)
        
        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF6600")
        goat = Dot(color="#FF6600").scale(1.5)
        
        # Path along curve
        path = axes.plot(lambda x: 0.5 * x**2 + 1)
        self.place_at_grid(ruler, 'B6', scale_factor=0.4)
        self.play(MoveAlongPath(goat, path), FadeIn(ruler), run_time=2)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        point_p = Dot(color="#FF0000")
        self.place_at_grid(point_p, 'C3', scale_factor=0.6) # Fix for Issue 25/37
        
        label_p = Text("P", font_size=20, color="#FF0000")
        self.place_at_grid(label_p, 'C4', scale_factor=0.5) # Fix for Issue 26/37
        
        self.place_at_grid(protractor, 'D4', scale_factor=0.5)
        
        question_mark = Text("?", font_size=30, color="#FF0000").next_to(label_p, UP)
        
        self.play(Create(point_p), Write(label_p), Flash(point_p, color="#FF0000"), FadeIn(protractor))
        self.play(Write(question_mark))
        self.play(FadeOut(question_mark))
        self.wait(1)
