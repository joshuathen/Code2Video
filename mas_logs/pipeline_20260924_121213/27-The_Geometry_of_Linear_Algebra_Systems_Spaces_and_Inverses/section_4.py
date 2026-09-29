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
        self.setup_layout("Inverse Matrices: The Reversal Process", [
            "Inverse matrices reverse the transformation.",
            "They map points back to original coordinates.",
            "Requires space not to be crushed."
        ])
        
        # === Animation for Lecture Line 1 ===
        vec_x = Arrow(start=self.grid['D3'], end=self.grid['D4'], color=BLUE)
        label_x = MathTex("x").next_to(vec_x.get_start(), LEFT)
        label_ax = MathTex("Ax").next_to(vec_x.get_end(), RIGHT)
        self.play(Create(vec_x), Write(label_x), Write(label_ax))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Using a simple line instead of updating attributes dynamically to avoid potential issues
        vec_inv = DashedLine(start=self.grid['D4'], end=self.grid['D3'], color="#32CD32")
        label_inv = MathTex("A^{-1}", color="#32CD32").next_to(vec_inv.get_center(), UP)
        self.play(Create(vec_inv), Write(label_inv))
        self.lecture[1].set_color("#32CD32")

        # === Animation for Lecture Line 3 ===
        crushed_line = Line(self.grid['E2'], self.grid['E5'], color=GREY)
        label_singular = Text("Singular: Space Crushed", font_size=20, color=GREY).next_to(crushed_line, DOWN)
        
        # Asset usage
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg"
        # We assume the file exists based on instructions. Use a simple shape if SVGMobject fails during runtime.
        try:
            icon = SVGMobject(asset_path).set_color(GREY)
        except Exception:
            icon = Circle(radius=0.2, color=GREY)
            
        self.place_at_grid(icon, 'F3', scale_factor=0.5)
        
        self.play(Create(crushed_line), Write(label_singular), FadeIn(icon))
        
        # Update colors instead of animation to maintain constraints
        vec_inv.set_color(GREY)
        label_inv.set_color(GREY)
        self.lecture[2].set_color(GREY)
        
        self.wait(2)
