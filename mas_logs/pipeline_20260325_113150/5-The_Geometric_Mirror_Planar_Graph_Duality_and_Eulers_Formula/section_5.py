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
        # Initial layout setup
        title = "Application: Duality in Design and Problem Solving"
        lecture_lines = [
            "Graph duality applies to practical designs like building floor plans.",
            "Walls represent the original graph, while rooms become dual vertices.",
            "Navigating the dual graph solves pathfinding through the original maze."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors as specified
        WALL_COLOR = "#00FF00"
        DUAL_COLOR = "#00FFFF"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Floor plan / Maze (The Original Graph)
        # Defining a 2x2 room structure centered at origin before anchoring
        floor_plan = VGroup(
            Line([-1, 1, 0], [1, 1, 0]), # Top wall
            Line([1, 1, 0], [1, -1, 0]), # Right wall
            Line([1, -1, 0], [-1, -1, 0]), # Bottom wall
            Line([-1, -1, 0], [-1, 1, 0]), # Left wall
            Line([0, 1, 0], [0, -1, 0]), # Vertical Divider
            Line([-1, 0, 0], [1, 0, 0]), # Horizontal Divider
        ).set_color(WALL_COLOR).set_stroke(width=8)
        
        # Use visual anchor system (Issue 33)
        self.place_in_area(floor_plan, 'B1', 'E3', scale_factor=0.8)
        
        self.play(Create(floor_plan), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(HIGHLIGHT_COLOR))
        
        # Dual vertices and edges (The rooms and their connections)
        # Coordinates relative to local origin before place_in_area
        dual_nodes = VGroup(
            Dot([-0.5, 0.5, 0], color=DUAL_COLOR, radius=0.12), # Top-Left Room
            Dot([0.5, 0.5, 0], color=DUAL_COLOR, radius=0.12),  # Top-Right Room
            Dot([0.5, -0.5, 0], color=DUAL_COLOR, radius=0.12), # Bottom-Right Room
            Dot([-0.5, -0.5, 0], color=DUAL_COLOR, radius=0.12), # Bottom-Left Room
        )
        
        # Connect dual nodes that share a wall
        dual_edges = VGroup(
            Line(dual_nodes[0].get_center(), dual_nodes[1].get_center(), color=DUAL_COLOR),
            Line(dual_nodes[1].get_center(), dual_nodes[2].get_center(), color=DUAL_COLOR),
            Line(dual_nodes[2].get_center(), dual_nodes[3].get_center(), color=DUAL_COLOR),
            Line(dual_nodes[3].get_center(), dual_nodes[0].get_center(), color=DUAL_COLOR),
        )
        
        dual_graph = VGroup(dual_nodes, dual_edges)
        
        # Anchor dual graph to the same area (Issue 34)
        self.place_in_area(dual_graph, 'B1', 'E3', scale_factor=0.8)
        
        self.play(FadeIn(dual_nodes, shift=UP*0.3), run_time=1)
        self.play(Create(dual_edges), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(HIGHLIGHT_COLOR))
        
        # Navigator (Pathfinding indicator)
        path_dot = Dot(color=WHITE, radius=0.15).set_z_index(10)
        
        # Anchor path dot to visual system (Issue 35)
        self.place_at_grid(path_dot, 'B2', scale_factor=0.6)
        
        self.play(FadeIn(path_dot))
        
        # Visualize the path taken in the dual graph
        p1 = Line(dual_nodes[0].get_center(), dual_nodes[1].get_center(), color=HIGHLIGHT_COLOR, stroke_width=6)
        p2 = Line(dual_nodes[1].get_center(), dual_nodes[2].get_center(), color=HIGHLIGHT_COLOR, stroke_width=6)
        
        # Navigation sequence
        # Snap to the actual dual node center then traverse
        self.play(path_dot.animate.move_to(dual_nodes[0]), run_time=0.8)
        self.play(
            path_dot.animate.move_to(dual_nodes[1]),
            Create(p1),
            run_time=1.2
        )
        self.play(
            path_dot.animate.move_to(dual_nodes[2]),
            Create(p2),
            run_time=1.2
        )
        
        # Final visualization: Emphasize the Geometric Mirror
        self.play(
            Indicate(floor_plan, color=WALL_COLOR, scale_factor=1.05),
            Indicate(dual_graph, color=DUAL_COLOR, scale_factor=1.05),
            run_time=2
        )
        
        self.wait(3)
