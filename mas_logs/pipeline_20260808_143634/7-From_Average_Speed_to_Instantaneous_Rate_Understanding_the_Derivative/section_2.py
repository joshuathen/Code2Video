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
        self.setup_layout("Visualizing the Slope of a Secant Line", [
            "Plot position versus time as a curve.", 
            "Connect two points with a secant line.", 
            "The slope shows average rate of change."
        ])
        
        # Set up graph area
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 5, 1], axis_config={"include_tip": False})
        self.place_in_area(axes, "B4", "F6", scale_factor=0.55)
        
        # === Animation for Lecture Line 1 ===
        curve = axes.plot(lambda x: 0.2 * x**2, color=WHITE)
        self.play(Create(curve))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        p1 = axes.coords_to_point(1, 0.2)
        p2 = axes.coords_to_point(4, 3.2)
        dot1 = Dot(p1, color="#FFD700")
        dot2 = Dot(p2, color="#FFD700")
        secant = Line(p1, p2, color="#32CD32")
        
        self.play(FadeIn(dot1), FadeIn(dot2))
        self.play(Create(secant))
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        label = MathTex(r"Slope = \frac{\Delta y}{\Delta x}", color="#32CD32", font_size=36)
        self.place_at_grid(label, "B2", scale_factor=0.4)
        self.play(Write(label))
        self.lecture[2].set_color("#32CD32")
        
        self.wait(2)
