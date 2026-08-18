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
        self.setup_layout("Backpropagation: The Credit Assignment Problem", [
            "Backpropagation calculates credit for error.",
            "We traverse backward using the chain rule.",
            "Each weight is updated by its contribution."
        ])

        # Assets: Load nodes
        node_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/node.svg"
        node_out = SVGMobject(node_path, color=WHITE, fill_opacity=0.5)
        node_mid = SVGMobject(node_path, color=WHITE, fill_opacity=0.5)
        node_in = SVGMobject(node_path, color=WHITE, fill_opacity=0.5)
        
        # Fixing positions based on issue 35
        self.place_at_grid(node_out, 'B5', scale_factor=0.5)
        self.place_at_grid(node_mid, 'C4', scale_factor=0.5)
        self.place_at_grid(node_in, 'D3', scale_factor=0.5)
        
        lines = VGroup(
            Line(node_in.get_right(), node_mid.get_left()),
            Line(node_mid.get_right(), node_out.get_left())
        )
        self.add(lines, node_in, node_mid, node_out)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF4500"))
        # Fixed error label position based on issue 36
        error_label = Text("Error", font_size=20, color="#FF4500")
        error_label.next_to(node_out, UP)
        # Apply scaling constraint B020: 0.7-0.8 for text
        error_label.scale(0.7)
        self.play(Create(error_label), node_out.animate.set_fill("#FF4500", opacity=0.8))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        
        arrow_back1 = Arrow(start=node_out.get_left(), end=node_mid.get_right(), color="#FFD700")
        arrow_back2 = Arrow(start=node_mid.get_left(), end=node_in.get_right(), color="#FFD700")
        
        self.play(GrowArrow(arrow_back1), GrowArrow(arrow_back2))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.play(
            node_mid.animate.set_fill("#FFD700", opacity=0.8),
            node_in.animate.set_fill("#FFD700", opacity=0.8)
        )
        self.wait(1)
