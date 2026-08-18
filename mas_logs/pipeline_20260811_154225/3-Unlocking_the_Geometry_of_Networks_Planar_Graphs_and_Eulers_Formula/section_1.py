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
        self.setup_layout("Prerequisites: What is a Planar Graph?", [
            "Graphs connect vertices with edges.",
            "Planar graphs avoid edge crossings.",
            "A tangled graph can be untangled."
        ])
        
        # Paths to assets
        node_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/node.svg"
        edge_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/edge.svg"
        
        # === Animation for Lecture Line 1 ===
        # Use SVG assets as requested
        self.lecture[0].set_color("#FFFFFF")
        node1 = SVGMobject(node_svg, color="#FFFFFF")
        node2 = SVGMobject(node_svg, color="#FFFFFF")
        self.place_at_grid(node1, "B4", scale_factor=0.5)
        self.place_at_grid(node2, "B6", scale_factor=0.5)
        edge = Line(node1.get_center(), node2.get_center(), color="#FFFFFF")
        
        self.play(FadeIn(node1), FadeIn(node2), Create(edge))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33FF57")
        # Highlighting as requested
        node1_h = node1.copy().set_color("#FF5733")
        node2_h = node2.copy().set_color("#FF5733")
        edge_h = edge.copy().set_color("#33FF57")
        
        self.play(Transform(node1, node1_h), Transform(node2, node2_h), Transform(edge, edge_h))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#33FF57")
        # Final planar graph settles
        node3 = SVGMobject(node_svg, color="#FF5733")
        self.place_at_grid(node3, "D5", scale_factor=0.5)
        edge2 = Line(node2.get_center(), node3.get_center(), color="#33FF57")
        edge3 = Line(node3.get_center(), node1.get_center(), color="#33FF57")
        
        self.play(FadeIn(node3), Create(edge2), Create(edge3))
        self.wait(1)
