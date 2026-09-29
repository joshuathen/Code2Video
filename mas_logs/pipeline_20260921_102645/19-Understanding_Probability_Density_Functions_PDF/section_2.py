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
        lecture_lines = ["The PDF is a density function.", "Values must always be positive.", "Total area under curve is one."]
        self.setup_layout("Defining the PDF Curve", lecture_lines)
        
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1.2, 0.5],
            axis_config={"include_tip": True, "color": WHITE}
        )
        # Fix 21: Axes sizing/position
        self.place_in_area(axes, 'D2', 'F6', scale_factor=0.4)
        
        # Bell curve
        curve = axes.plot(lambda x: np.exp(-x**2), x_range=[-3, 3], color="#FFD700")
        # Fix 23: Curve sizing/position
        self.place_in_area(curve, 'D2', 'F6', scale_factor=0.4)
        
        # Area under curve
        area = axes.get_area(curve, x_range=[-3, 3], color="#FFD700", opacity=0.3)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.add(axes)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        self.play(Create(curve))
        
        # Label f(x)
        label = Text("f(x)", color="#FFD700", font_size=24)
        # Fix 22: Label positioning
        self.place_at_grid(label, 'D1', scale_factor=0.6)
        self.add(label)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        self.play(FadeIn(area))
        
        self.wait(2)
