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
        self.setup_layout("The Integrated Mapping", [
            "Triangle links b, x, and y.",
            "Vertex inputs shift the operation.",
            "Powers find the resulting y.",
            "Roots find the base b.",
            "Logs find the exponent x."
        ])
        
        # Setup Triangle and Labels
        p1 = self.grid["B3"] # top: y
        p2 = self.grid["E1"] # left: b
        p3 = self.grid["E5"] # right: x
        
        triangle = Polygon(p1, p2, p3, color="#FFD700")
        label_y = MathTex("y").next_to(p1, UP)
        label_b = MathTex("b").next_to(p2, LEFT)
        label_x = MathTex("x").next_to(p3, RIGHT)
        
        # Load Assets
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        self.place_at_grid(protractor, "A6", scale_factor=0.5)
        self.place_at_grid(ruler, "F6", scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(triangle), Write(label_y), Write(label_b), Write(label_x))
        self.lecture[0].set_color("#FFD700")
        
        # === Animation for Lecture Line 2 ===
        self.play(Rotate(triangle, angle=PI/6, about_point=triangle.get_center()))
        self.lecture[1].set_color("#87CEEB")
        
        # === Animation for Lecture Line 3 ===
        edge_by = Line(p2, p1, color="#FF4500", stroke_width=6)
        self.play(Create(edge_by))
        self.lecture[2].set_color("#FF4500")
        
        # === Animation for Lecture Line 4 ===
        edge_yb = Line(p1, p2, color="#32CD32", stroke_width=6)
        self.play(Create(edge_yb))
        self.lecture[3].set_color("#32CD32")
        
        # === Animation for Lecture Line 5 ===
        edge_bx = Line(p2, p3, color="#FFFFFF", stroke_width=6)
        self.play(Create(edge_bx))
        self.lecture[4].set_color("#FFFFFF")
        
        self.wait(2)
