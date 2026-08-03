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
        self.setup_layout(
            "The Bridge Construction (Transformation Rules)", 
            [
                "First, place a new vertex inside every original face.",
                "Don't forget the vertex for the infinite outer face.",
                "Connect dual vertices if their faces share an edge.",
                "Each original edge is crossed by exactly one dual edge.",
                "The resulting structure is the original's dual graph."
            ]
        )

        # Colors
        ORIGINAL_COLOR = GREY_B
        DUAL_COLOR = "#FF0000"  # Red
        HIGHLIGHT_COLOR = YELLOW

        # Define Cube Projection (Planar Graph)
        # Original Vertices
        v_o1 = self.grid["B2"]
        v_o2 = self.grid["B5"]
        v_o3 = self.grid["E5"]
        v_o4 = self.grid["E2"]
        v_i1 = self.grid["C3"]
        v_i2 = self.grid["C4"]
        v_i3 = self.grid["D4"]
        v_i4 = self.grid["D3"]

        orig_vertices = VGroup(*[Dot(pos, color=ORIGINAL_COLOR) for pos in [v_o1, v_o2, v_o3, v_o4, v_i1, v_i2, v_i3, v_i4]])
        
        orig_edges_def = [
            (v_o1, v_o2), (v_o2, v_o3), (v_o3, v_o4), (v_o4, v_o1), # Outer Square
            (v_i1, v_i2), (v_i2, v_i3), (v_i3, v_i4), (v_i4, v_i1), # Inner Square
            (v_o1, v_i1), (v_o2, v_i2), (v_o3, v_i3), (v_o4, v_i4)  # Connectors
        ]
        orig_edges = VGroup(*[Line(start, end, color=ORIGINAL_COLOR, stroke_width=2) for start, end in orig_edges_def])
        original_graph = VGroup(orig_edges, orig_vertices)

        # Dual Vertices (using place_in_area to ensure they are inside faces)
        # Define positions by area centers
        d_inner_pos = self.grid["D4"] # actually center of C3-D4
        d_inner = Dot(color=DUAL_COLOR); self.place_in_area(d_inner, "C3", "D4")
        d_top = Dot(color=DUAL_COLOR); self.place_in_area(d_top, "B3", "C4")
        d_bottom = Dot(color=DUAL_COLOR); self.place_in_area(d_bottom, "D3", "E4")
        d_left = Dot(color=DUAL_COLOR); self.place_in_area(d_left, "C2", "D3")
        d_right = Dot(color=DUAL_COLOR); self.place_in_area(d_right, "C4", "D5")
        d_outer = Dot(color=DUAL_COLOR); self.place_at_grid(d_outer, "A6")

        dual_dots_dict = {
            "inner": d_inner,
            "top": d_top,
            "bottom": d_bottom,
            "left": d_left,
            "right": d_right,
            "outer": d_outer
        }
        dual_dots = VGroup(*dual_dots_dict.values())
        inner_dots = VGroup(*[dual_dots_dict[k] for k in ["inner", "top", "bottom", "left", "right"]])
        outer_dot = dual_dots_dict["outer"]
        
        v_star_label = Text("V*", color=DUAL_COLOR, font_size=18)
        v_star_label.next_to(outer_dot, UP, buff=0.1)

        # Dual Edges
        dual_edges_list = [
            (dual_dots_dict["inner"], dual_dots_dict["top"]),
            (dual_dots_dict["inner"], dual_dots_dict["right"]),
            (dual_dots_dict["inner"], dual_dots_dict["bottom"]),
            (dual_dots_dict["inner"], dual_dots_dict["left"]),
            (dual_dots_dict["outer"], dual_dots_dict["top"]),
            (dual_dots_dict["outer"], dual_dots_dict["right"]),
            (dual_dots_dict["outer"], dual_dots_dict["bottom"]),
            (dual_dots_dict["outer"], dual_dots_dict["left"]),
            (dual_dots_dict["top"], dual_dots_dict["left"]),
            (dual_dots_dict["top"], dual_dots_dict["right"]),
            (dual_dots_dict["bottom"], dual_dots_dict["left"]),
            (dual_dots_dict["bottom"], dual_dots_dict["right"])
        ]
        dual_edges = VGroup(*[Line(s.get_center(), e.get_center(), color=DUAL_COLOR, stroke_width=2) for s, e in dual_edges_list])

        # Intersection points for flashing (Animation 4)
        flash_points = [edge.get_center() for edge in dual_edges]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        self.play(Create(original_graph))
        self.wait(0.5)
        self.play(Create(inner_dots), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        self.play(Create(outer_dot), Write(v_star_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(DUAL_COLOR)
        self.play(Create(dual_edges), run_time=2.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(DUAL_COLOR)
        # Pick intersections to flash
        self.play(
            *[Flash(p, color=HIGHLIGHT_COLOR, line_length=0.2, flash_radius=0.3) for p in flash_points],
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GREEN)
        
        # Issue 29 Fix: Use place_in_area for label
        label = Text("Dual Graph (Octahedron)", font_size=20, color=GREEN)
        self.place_in_area(label, 'A2', 'A5', scale_factor=0.8)
        
        self.play(
            original_graph.animate.set_opacity(0.2),
            dual_edges.animate.set_color(GREEN),
            dual_dots.animate.set_color(GREEN),
            v_star_label.animate.set_color(GREEN),
            Write(label)
        )
        self.wait(2)
