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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion: A New Mathematical Reality", [
            "Changing distance metrics shifts convergence.",
            "2-adic analysis reveals deep hidden structure.",
            "This is vital for modern number theory."
        ])
        
        # Euclidean line
        euclid_line = NumberLine(x_range=[-2, 2], length=4, color=BLUE)
        euclid_label = Text("Euclidean (R)", font_size=20, color=BLUE).scale(0.7)
        euclid_label.next_to(euclid_line, UP)
        
        # 2-adic line (simulated via Cantor set-like structure)
        q2_line = VGroup(*[Dot(point=RIGHT*i*0.2, color=RED) for i in range(-10, 11)])
        q2_label = Text("2-adic (Q2)", font_size=20, color=RED).scale(0.7)
        q2_label.next_to(q2_line, DOWN)

        # Assets
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        asset_1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg").set_color(WHITE)
        asset_2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg").set_color("#00FFFF")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        self.place_at_grid(euclid_line, 'B4', scale_factor=0.9)
        self.place_at_grid(euclid_label, 'B4', scale_factor=0.7)
        # Position asset using the grid area approach
        self.place_at_grid(asset_1, 'A5', scale_factor=0.5)
        
        self.play(Create(euclid_line), Write(euclid_label), FadeIn(asset_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00")
        
        self.place_at_grid(q2_line, 'E4', scale_factor=0.9)
        self.place_at_grid(q2_label, 'E4', scale_factor=0.7)
        self.place_at_grid(asset_2, 'F5', scale_factor=0.5)
        
        self.play(Create(q2_line), Write(q2_label), FadeIn(asset_2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        
        # Zoom out all
        zoom_group = VGroup(euclid_line, q2_line, asset_1, asset_2)
        self.play(zoom_group.animate.scale(0.8))
        self.wait(2)
