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
        title = "The Golden Rule: Euler's Characteristic Formula"
        lines = [
            "Euler's formula states V minus E plus F equals 2.",
            "This balance holds for all connected planar graphs.",
            "Adding a tail vertex keeps the sum constant.",
            "Adding an edge creates exactly one new face.",
            "The formula remains invariant during these transformations."
        ]
        self.setup_layout(title, lines)

        # Helper to create vertices and edges
        GRAPH_COLOR = "#FF8C00"
        HIGHLIGHT_COLOR = "#FFFF00"

        # Define specific grid points for the graph
        # Centering the graph slightly better in the B-E rows
        v1_pos = self.grid["B3"]
        v2_pos = self.grid["D2"]
        v3_pos = self.grid["D4"]
        v4_pos = self.grid["E4"]
        v5_pos = self.grid["F4"]

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Issue 22 Fix: place_in_area formula in A2-A5
        formula = MathTex("V", "-", "E", "+", "F", "=", "2", font_size=48)
        self.place_in_area(formula, "A2", "A5")
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )

        # Create triangle graph with a tail
        # Vertices
        dots = VGroup(
            Dot(v1_pos, color=GRAPH_COLOR),
            Dot(v2_pos, color=GRAPH_COLOR),
            Dot(v3_pos, color=GRAPH_COLOR),
            Dot(v4_pos, color=GRAPH_COLOR)
        )
        # Edges
        edges = VGroup(
            Line(v1_pos, v2_pos, color=GRAPH_COLOR),
            Line(v2_pos, v3_pos, color=GRAPH_COLOR),
            Line(v3_pos, v1_pos, color=GRAPH_COLOR),
            Line(v3_pos, v4_pos, color=GRAPH_COLOR)
        )
        
        self.play(Create(dots), Create(edges))

        # Labels for V, E, F
        # Issue 23 Fix: place_at_grid counts_label at F3, scale 0.9
        counts_label = MathTex("V=4, E=4, F=2", font_size=36, color=WHITE)
        self.place_at_grid(counts_label, "F3", scale_factor=0.9)
        
        # Issue 24 Fix: place_at_grid check_formula at F4, scale 0.9
        # Also splitting for the pulse animation later
        check_formula = MathTex("4", "-", "4", "+", "2", "=", "2", font_size=36, color=WHITE)
        self.place_at_grid(check_formula, "F4", scale_factor=0.9)

        self.play(Write(counts_label), Write(check_formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )

        # Add a new vertex and edge to the tail (v5 connected to v4)
        new_dot = Dot(v5_pos, color=GRAPH_COLOR)
        new_edge = Line(v4_pos, v5_pos, color=GRAPH_COLOR)
        
        self.play(FadeIn(new_dot), Create(new_edge))
        
        # Update counts: V=5, E=5, F=2
        # Issue 23 Fix: place_at_grid new_counts at F3, scale 0.9
        new_counts = MathTex("V=5, E=5, F=2", font_size=36, color=WHITE)
        self.place_at_grid(new_counts, "F3", scale_factor=0.9)
        
        # Issue 24 Fix: place_at_grid new_check at F4, scale 0.9
        new_check = MathTex("5", "-", "5", "+", "2", "=", "2", font_size=36, color=WHITE)
        self.place_at_grid(new_check, "F4", scale_factor=0.9)

        self.play(
            Transform(counts_label, new_counts),
            Transform(check_formula, new_check)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight lecture line
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HIGHLIGHT_COLOR)
        )

        # Add an edge between v2 and v4 to form a new cycle
        cycle_edge = Line(v2_pos, v4_pos, color=GRAPH_COLOR)
        
        # Indicator for a new face
        face_indicator = Polygon(v2_pos, v3_pos, v4_pos, fill_opacity=0.3, color=GRAPH_COLOR, stroke_width=0)
        
        self.play(Create(cycle_edge))
        self.play(FadeIn(face_indicator))
        self.play(FadeOut(face_indicator))

        # Update counts: V=5, E=6, F=3
        # Issue 23 Fix: place_at_grid new_counts_4 at F3, scale 0.9
        new_counts_4 = MathTex("V=5, E=6, F=3", font_size=36, color=WHITE)
        self.place_at_grid(new_counts_4, "F3", scale_factor=0.9)
        
        # Issue 24 Fix: place_at_grid new_check_4 at F4, scale 0.9
        new_check_4 = MathTex("5", "-", "6", "+", "3", "=", "2", font_size=36, color=WHITE)
        self.place_at_grid(new_check_4, "F4", scale_factor=0.9)

        self.play(
            Transform(counts_label, new_counts_4),
            Transform(check_formula, new_check_4)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight lecture line
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT_COLOR)
        )

        # Flash the "= 2" part and pulse the graph
        all_graph_elements = VGroup(dots, edges, new_dot, new_edge, cycle_edge)
        
        # formula indices: V(0) -(1) E(2) +(3) F(4) =(5) 2(6)
        # check_formula indices: VAL(0) -(1) VAL(2) +(3) VAL(4) =(5) 2(6)
        
        self.play(
            formula[5:].animate.set_color(HIGHLIGHT_COLOR),
            check_formula[5:].animate.set_color(HIGHLIGHT_COLOR),
            all_graph_elements.animate.scale(1.1),
            run_time=0.5
        )
        self.play(
            formula[5:].animate.set_color(WHITE),
            check_formula[5:].animate.set_color(WHITE),
            all_graph_elements.animate.scale(1/1.1),
            run_time=0.5
        )
        
        self.wait(2)
