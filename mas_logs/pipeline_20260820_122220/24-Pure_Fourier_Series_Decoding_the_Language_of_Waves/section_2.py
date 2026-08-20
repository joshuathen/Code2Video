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
        lecture_lines = [
            "Sine and cosine functions are perfectly orthogonal.",
            "Like spatial axes, they don't interfere.",
            "This orthogonality isolates specific signal components."
        ]
        self.setup_layout("Prerequisite Refresher: Orthogonality", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Draw two perpendicular vectors labeled 'a' and 'b' in #FF4500
        vec_a = Vector([0, 1.5], color="#FF4500")
        vec_b = Vector([1.5, 0], color="#FF4500")
        label_a = MathTex(r"\vec{a}", color="#FF4500").next_to(vec_a.get_end(), UP)
        label_b = MathTex(r"\vec{b}", color="#FF4500").next_to(vec_b.get_end(), RIGHT)
        
        axes_group = VGroup(vec_a, vec_b, label_a, label_b)
        self.place_in_area(axes_group, 'C4', 'F6', scale_factor=0.6)
        
        self.play(Create(vec_a), Create(vec_b), Write(label_a), Write(label_b))
        self.lecture[0].set_color("#FF4500")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show dot product calculation equaling zero in #00FF00
        dot_product = MathTex(r"\vec{a} \cdot \vec{b} = 0", color="#00FF00")
        self.place_at_grid(dot_product, 'B4', scale_factor=0.9)
        
        self.play(Write(dot_product))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate projection of one vector onto another in #FFFF00
        proj_line = DashedLine(vec_a.get_end(), vec_b.get_start(), color="#FFFF00")
        proj_label = MathTex(r"\text{proj}", color="#FFFF00", font_size=24)
        self.place_at_grid(proj_label, 'E5', scale_factor=0.7)
        
        self.play(Create(proj_line), Write(proj_label))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
