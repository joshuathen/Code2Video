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

class Section3Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "Entering the Mirror Dimension: Concept of Duality"
        lecture_lines = [
            "Every planar graph has a hidden dual graph.",
            "Place a new vertex inside every face.",
            "Connect vertices if their faces share an edge.",
            "These new connections form the dual edges.",
            "The dual graph mirrors the original structure."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        PRIMAL_COLOR = "#FFFFFF"
        DUAL_COLOR = "#FF00FF"

        # === Animation for Lecture Line 1 ===
        # Display a simple "Primal" graph (a triangle with a line inside) in #FFFFFF.
        self.lecture[0].set_color(PRIMAL_COLOR)
        
        # Define vertices relative to origin for better centering control
        v1 = np.array([0, 1.5, 0])
        v2 = np.array([-1.5, -1, 0])
        v3 = np.array([1.5, -1, 0])
        v4 = np.array([0, 0, 0])
        
        p_v1 = Dot(v1, color=PRIMAL_COLOR)
        p_v2 = Dot(v2, color=PRIMAL_COLOR)
        p_v3 = Dot(v3, color=PRIMAL_COLOR)
        p_v4 = Dot(v4, color=PRIMAL_COLOR)
        
        p_e1 = Line(v1, v2, color=PRIMAL_COLOR)
        p_e2 = Line(v2, v3, color=PRIMAL_COLOR)
        p_e3 = Line(v3, v1, color=PRIMAL_COLOR)
        p_e4 = Line(v1, v4, color=PRIMAL_COLOR)
        p_e5 = Line(v2, v4, color=PRIMAL_COLOR)
        p_e6 = Line(v3, v4, color=PRIMAL_COLOR)
        
        primal_edges = VGroup(p_e1, p_e2, p_e3, p_e4, p_e5, p_e6)
        primal_dots = VGroup(p_v1, p_v2, p_v3, p_v4)
        primal_graph = VGroup(primal_edges, primal_dots)
        
        # Issue 25: centering the primal graph on the right side
        self.place_in_area(primal_graph, 'B2', 'F6', scale_factor=0.8)
        
        self.play(Create(primal_graph))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Place new vertices (Dual Vertices) in the center of every face in #FF00FF.
        self.lecture[1].set_color(DUAL_COLOR)
        
        # Capture positions after placement
        p1, p2, p3, p4 = [v.get_center() for v in [p_v1, p_v2, p_v3, p_v4]]
        
        # Calculate centroids for inner faces
        dv1_pos = (p1 + p2 + p4) / 3
        dv2_pos = (p1 + p3 + p4) / 3
        dv3_pos = (p2 + p3 + p4) / 3
        
        d_v1 = Dot(dv1_pos, color=DUAL_COLOR)
        d_v2 = Dot(dv2_pos, color=DUAL_COLOR)
        d_v3 = Dot(dv3_pos, color=DUAL_COLOR)
        
        # Issue 24: Place exterior dual vertex at A4 to avoid title/lecture overlap
        d_v4 = Dot(color=DUAL_COLOR)
        self.place_at_grid(d_v4, 'A4', scale_factor=1.0)
        dv4_pos = d_v4.get_center()
        
        dual_dots = VGroup(d_v1, d_v2, d_v3, d_v4)
        
        self.play(Create(dual_dots))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Connect vertices if their faces share an edge.
        self.lecture[2].set_color(DUAL_COLOR)
        
        # Internal Dual Edges (connecting vertices in adjacent inner faces)
        de1 = Line(dv1_pos, dv2_pos, color=DUAL_COLOR) # Crosses p_e4
        de2 = Line(dv1_pos, dv3_pos, color=DUAL_COLOR) # Crosses p_e5
        de3 = Line(dv2_pos, dv3_pos, color=DUAL_COLOR) # Crosses p_e6
        
        # Boundary Dual Edges (connecting inner faces to the outer face vertex)
        # Use arcs to wrap around boundary edges correctly
        de4 = ArcBetweenPoints(dv1_pos, dv4_pos, angle=PI/4, color=DUAL_COLOR)  # Crosses p_e1
        de5 = ArcBetweenPoints(dv3_pos, dv4_pos, angle=PI/1.2, color=DUAL_COLOR) # Crosses p_e2 (bottom edge)
        de6 = ArcBetweenPoints(dv2_pos, dv4_pos, angle=-PI/4, color=DUAL_COLOR) # Crosses p_e3
        
        dual_edges = VGroup(de1, de2, de3, de4, de5, de6)
        
        self.play(Create(dual_edges))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Fade out the original "Primal" graph edges and vertices, leaving only the "Dual" graph.
        self.lecture[3].set_color(DUAL_COLOR)
        
        self.play(FadeOut(primal_graph))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Pulse the entire Dual graph to show it is a complete, new network.
        self.lecture[4].set_color(DUAL_COLOR)
        
        dual_graph = VGroup(dual_dots, dual_edges)
        self.play(dual_graph.animate.scale(1.15), run_time=0.4)
        self.play(dual_graph.animate.scale(1/1.15), run_time=0.4)
        self.wait(2)
