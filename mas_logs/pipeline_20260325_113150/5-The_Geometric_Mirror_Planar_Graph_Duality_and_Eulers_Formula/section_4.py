from manim import *

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
        # Setup the layout with the title and lecture lines
        self.setup_layout("The Mathematical Symmetry: V-F Swapping", [
            "Consider the planar representation of a three-dimensional cube graph.",
            "The cube has eight vertices, twelve edges, and six faces.",
            "The dual graph has six vertices, matching the original faces.",
            "Both graphs share exactly twelve edges, showing a shared count.",
            "The dual has eight faces, equal to the original vertices."
        ])

        # Colors
        V_COLOR = "#00FF00"  # Original vertices
        E_COLOR = "#FFFFFF"  # Original edges
        V_DUAL_COLOR = "#FFA500"  # Dual vertices
        E_DUAL_COLOR = "#00FFFF"  # Dual edges

        # === Animation for Lecture Line 1 ===
        # Consider the planar representation of a three-dimensional cube graph.
        self.lecture[0].set_color(YELLOW)
        
        # Define vertices for a planar representation of a cube (inner square + outer square)
        # Outer square
        v1 = self.grid["B2"]
        v2 = self.grid["B5"]
        v3 = self.grid["E5"]
        v4 = self.grid["E2"]
        # Inner square
        v5 = self.grid["C3"]
        v6 = self.grid["C4"]
        v7 = self.grid["D4"]
        v8 = self.grid["D3"]
        
        orig_v_points = [v1, v2, v3, v4, v5, v6, v7, v8]
        orig_vertices = VGroup(*[Dot(p, color=V_COLOR) for p in orig_v_points])
        
        orig_edges_list = [
            (v1, v2), (v2, v3), (v3, v4), (v4, v1), # Outer
            (v5, v6), (v6, v7), (v7, v8), (v8, v5), # Inner
            (v1, v5), (v2, v6), (v3, v7), (v4, v8)  # Connectors (struts)
        ]
        orig_edges = VGroup(*[Line(start, end, color=E_COLOR) for start, end in orig_edges_list])
        
        self.play(Create(orig_vertices), Create(orig_edges))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The cube has eight vertices, twelve edges, and six faces.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Original stats
        stats_g = Text("Original G: V=8, E=12, F=6", font_size=24, color=WHITE)
        self.place_in_area(stats_g, "A1", "A6")
        self.play(Write(stats_g))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The dual graph has six vertices, matching the original faces.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(V_DUAL_COLOR)
        
        # Dual Vertices (one in each of the 6 faces)
        # Face locations
        f1_pos = self.place_in_area(Dot(), "C3", "D4").get_center() # Inner square face
        f2_pos = self.place_in_area(Dot(), "B3", "B4").get_center() # Top trapezoid face
        f3_pos = self.place_in_area(Dot(), "C5", "D5").get_center() # Right trapezoid face
        f4_pos = self.place_in_area(Dot(), "E3", "E4").get_center() # Bottom trapezoid face
        f5_pos = self.place_in_area(Dot(), "C2", "D2").get_center() # Left trapezoid face
        f6_pos = self.grid["F6"] # Exterior face
        
        dual_v_points = [f1_pos, f2_pos, f3_pos, f4_pos, f5_pos, f6_pos]
        dual_vertices = VGroup(*[Dot(p, color=V_DUAL_COLOR) for p in dual_v_points])
        
        label_v_dual = Text("V* = 6", font_size=24, color=V_DUAL_COLOR)
        self.place_at_grid(label_v_dual, "B1")
        
        self.play(FadeIn(dual_vertices), Write(label_v_dual))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Both graphs share exactly twelve edges, showing a shared count.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(E_DUAL_COLOR)
        
        # Dual Edges
        dual_edges_list = [
            (f1_pos, f2_pos), (f1_pos, f3_pos), (f1_pos, f4_pos), (f1_pos, f5_pos), # From inner face
            (f6_pos, f2_pos), (f6_pos, f3_pos), (f6_pos, f4_pos), (f6_pos, f5_pos), # From outer face
            (f5_pos, f2_pos), (f2_pos, f3_pos), (f3_pos, f4_pos), (f4_pos, f5_pos)  # Perimeter duals
        ]
        # Curved edges for the outer ones to make it look nicer
        dual_edges = VGroup()
        for i, (start, end) in enumerate(dual_edges_list):
            if i >= 4 and i <= 7: # External edges
                dual_edges.add(ArcBetweenPoints(start, end, color=E_DUAL_COLOR, radius=5))
            else:
                dual_edges.add(Line(start, end, color=E_DUAL_COLOR))
        
        label_e_dual = Text("E* = 12", font_size=24, color=E_DUAL_COLOR)
        # Issue 31: Move to C1 to fix vertical gap
        self.place_at_grid(label_e_dual, "C1")
        
        self.play(Create(dual_edges), Write(label_e_dual))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The dual has eight faces, equal to the original vertices.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(V_DUAL_COLOR)
        
        label_f_dual = Text("F* = 8", font_size=24, color=V_DUAL_COLOR)
        # Issue 32: Move to D1 to maintain list grouping
        self.place_at_grid(label_f_dual, "D1")
        
        # Highlight correlation
        v_orig_val = Text("(Original V = 8)", font_size=18, color=V_COLOR)
        # Issue 30: Move to E1 with scale 0.6 to avoid overlap with final summary at F1
        self.place_at_grid(v_orig_val, "E1", scale_factor=0.6)
        
        # Final Summary Table
        summary_dual = Text("Dual G*: V*=6, E*=12, F*=8", font_size=24, color=V_DUAL_COLOR)
        self.place_in_area(summary_dual, "F1", "F6")
        
        self.play(
            Write(label_f_dual),
            Write(v_orig_val),
            Write(summary_dual),
            orig_edges.animate.set_stroke(opacity=0.2),
            orig_vertices.animate.set_fill(opacity=0.2)
        )
        self.wait(2)
