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
        self.setup_layout("Prerequisite Review: The Two Pillars", [
            "Differentiation measures the slope of a curve.", 
            "Integration calculates the total area under a curve.", 
            "Velocity is the derivative of position."
        ])
        
        # Create assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        graph = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        car = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        
        # Create pillars/containers for assets
        derivative_pillar = Rectangle(height=1.5, width=1.5, color=WHITE, fill_opacity=0.3)
        integral_pillar = Rectangle(height=1.5, width=1.5, color=WHITE, fill_opacity=0.3)
        
        d_label = Text("Slope", font_size=24)
        i_label = Text("Area", font_size=24)
        
        # Applying layouts based on feedback
        self.place_in_area(derivative_pillar, 'D1', 'D2', scale_factor=0.8)
        self.place_in_area(integral_pillar, 'D4', 'D5', scale_factor=0.8)
        self.place_at_grid(d_label, 'B2', scale_factor=0.5)
        self.place_at_grid(i_label, 'B5', scale_factor=0.5)
        self.place_at_grid(ruler, 'D2', scale_factor=0.5)
        self.place_at_grid(graph, 'D5', scale_factor=0.5)
        self.place_at_grid(car, 'F3', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(derivative_pillar), FadeIn(d_label), FadeIn(ruler))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(integral_pillar), FadeIn(i_label), FadeIn(graph))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(car))
        self.lecture[2].set_color("#FF00FF")
        self.wait(2)
