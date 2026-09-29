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
        self.setup_layout("Prerequisite: The Projection Intuition", ["Dot product measures vector alignment.", "Vectors project as shadows.", "Projection length is the dot product."])
        
        # Create elements
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/flashlight.svg]
        flashlight = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/flashlight.svg")
        
        v = Vector([1.5, 1.0], color=WHITE)
        b = Vector([2.0, 0], color=BLUE)
        
        # Projection math
        dot_vb = v.get_end()[0] * b.get_end()[0] + v.get_end()[1] * b.get_end()[1]
        dot_bb = b.get_end()[0]**2 + b.get_end()[1]**2
        scalar = dot_vb / dot_bb
        p_end = scalar * b.get_end()
        p = Vector(p_end, color="#FFD700")
        
        ortho_comp = Vector(v.get_end() - p.get_end(), color="#FF6347").shift(p.get_end())
        
        # Group and position
        vector_group = VGroup(v, b, p, ortho_comp)
        self.place_at_grid(vector_group, 'C4', scale_factor=0.8)
        self.place_at_grid(flashlight, 'B2', scale_factor=0.3)
        
        # Labels
        v_label = Text("v", font_size=20, color=WHITE).next_to(v.get_end(), UP, buff=0.1)
        p_label = Text("p", font_size=20, color="#FFD700").next_to(p.get_end(), DOWN, buff=0.1)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(v), FadeIn(flashlight))
        self.lecture[0].set_color("#FFD700")
        self.play(Write(v_label))

        # === Animation for Lecture Line 2 ===
        self.play(Create(p), Create(ortho_comp))
        self.lecture[1].set_color("#FFD700")
        self.play(Write(p_label))
        
        # Highlight right angle
        angle = RightAngle(Line(ORIGIN, p.get_end()), Line(p.get_end(), v.get_end()), length=0.2)
        self.play(Create(angle))

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(ortho_comp), FadeOut(angle))
        self.lecture[2].set_color("#FFD700")
        self.wait(2)
