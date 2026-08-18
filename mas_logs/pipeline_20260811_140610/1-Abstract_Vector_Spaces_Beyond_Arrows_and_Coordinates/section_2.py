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
        self.setup_layout(
            "Defining the 'Rules of the Game'", 
            ["Closure ensures operations stay within the set.", 
             "Adding two vectors must yield another vector.", 
             "Scaling a vector must also remain inside."]
        )
        
        # Set V and vectors
        set_v = Circle(radius=1.2, color=WHITE)
        self.place_at_grid(set_v, 'C4')
        
        v1 = Dot(color="#FF5733")
        v2 = Dot(color="#FF5733")
        self.place_at_grid(v1, 'C4')
        self.place_at_grid(v2, 'D5')
        
        # Labels
        v_label = MathTex("v", color="#33FF57")
        w_label = MathTex("w", color="#33FF57")
        c_label = MathTex("c", color="#3357FF")
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(set_v), FadeIn(v1), FadeIn(v2))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        # Re-using the vector diagram
        vec_diagram = VGroup(v1, v2)
        self.place_in_area(vec_diagram, 'B3', 'E5', scale_factor=0.9)
        
        # Fix labels per requirement
        self.place_at_grid(v_label, 'B2', scale_factor=0.7)
        self.place_at_grid(w_label, 'E5', scale_factor=0.7)
        
        self.play(
            FadeIn(v_label), FadeIn(w_label), 
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#33FF57")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fix labels per requirement
        self.place_at_grid(c_label, 'F4', scale_factor=0.7)
        
        # Grid table visual placeholder
        grid_table = Rectangle(width=2.5, height=2.5, color=WHITE)
        self.place_in_area(grid_table, 'B2', 'F6', scale_factor=0.85)
        
        self.play(
            FadeIn(c_label), Create(grid_table),
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#3357FF")
        )
        self.wait(2)
