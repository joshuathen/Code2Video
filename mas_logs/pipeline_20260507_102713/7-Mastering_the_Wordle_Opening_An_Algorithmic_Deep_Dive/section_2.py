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
        # Initialize Scene layout
        lecture_lines = [
            'Entropy measures how much uncertainty we reduce.',
            'Information gain splits our pool of possible answers.',
            'One bit of information cuts the possibilities in half.'
        ]
        self.setup_layout("Prerequisite: Understanding Entropy and Information Theory", lecture_lines)
        
        HIGHLIGHT_COLOR = "#5865F2" # Blue
        NODE_COLOR = "#FFFFFF" # White

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Root node using asset [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/words.svg]
        root_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/words.svg", color=NODE_COLOR)
        root_label = Text("Total Words", font_size=16, color=NODE_COLOR)
        self.place_in_area(root_svg, "A3", "A4", scale_factor=0.6)
        root_label.next_to(root_svg, DOWN, buff=0.1)
        
        root_node_group = VGroup(root_svg, root_label)
        
        self.play(DrawBorderThenFill(root_svg), Write(root_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        # Level 1 Nodes - Issue 31 Fix
        node_l1_1 = Circle(radius=0.25, color=NODE_COLOR, fill_opacity=0.2)
        node_l1_2 = Circle(radius=0.25, color=NODE_COLOR, fill_opacity=0.2)
        self.place_in_area(node_l1_1, "C2", "C3")
        self.place_in_area(node_l1_2, "C4", "C5")
        
        # Level 2 Nodes - Issue 32 Fix
        node_l2_1 = Circle(radius=0.18, color=NODE_COLOR, fill_opacity=0.2)
        node_l2_2 = Circle(radius=0.18, color=NODE_COLOR, fill_opacity=0.2)
        node_l2_3 = Circle(radius=0.18, color=NODE_COLOR, fill_opacity=0.2)
        node_l2_4 = Circle(radius=0.18, color=NODE_COLOR, fill_opacity=0.2)
        self.place_at_grid(node_l2_1, "E2", scale_factor=0.8)
        self.place_at_grid(node_l2_2, "E3", scale_factor=0.8)
        self.place_at_grid(node_l2_3, "E4", scale_factor=0.8)
        self.place_at_grid(node_l2_4, "E5", scale_factor=0.8)
        
        # Edges
        edge1 = Line(root_svg.get_bottom(), node_l1_1.get_top(), color=NODE_COLOR)
        edge2 = Line(root_svg.get_bottom(), node_l1_2.get_top(), color=NODE_COLOR)
        
        edge_l2_1 = Line(node_l1_1.get_bottom(), node_l2_1.get_top(), color=NODE_COLOR)
        edge_l2_2 = Line(node_l1_1.get_bottom(), node_l2_2.get_top(), color=NODE_COLOR)
        edge_l2_3 = Line(node_l1_2.get_bottom(), node_l2_3.get_top(), color=NODE_COLOR)
        edge_l2_4 = Line(node_l1_2.get_bottom(), node_l2_4.get_top(), color=NODE_COLOR)
        
        tree_edges = VGroup(edge1, edge2, edge_l2_1, edge_l2_2, edge_l2_3, edge_l2_4)
        tree_circles = VGroup(node_l1_1, node_l1_2, node_l2_1, node_l2_2, node_l2_3, node_l2_4)
        
        self.play(
            Create(tree_edges),
            Create(tree_circles)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(HIGHLIGHT_COLOR))
        
        # Highlight path
        path_highlight = VGroup(edge1, node_l1_1, edge_l2_2, node_l2_2)
        
        # Issue 33 Fix: Reposition bit_label
        bit_label = Text("1 Bit Reduction", font_size=18, color=HIGHLIGHT_COLOR)
        self.place_in_area(bit_label, "F2", "F3")
        
        self.play(
            path_highlight.animate.set_color(HIGHLIGHT_COLOR).set_stroke(width=6),
            Write(bit_label)
        )
        self.wait(2)
