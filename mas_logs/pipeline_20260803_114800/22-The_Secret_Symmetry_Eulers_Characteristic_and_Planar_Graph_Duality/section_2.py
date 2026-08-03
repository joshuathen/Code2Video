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
        self.setup_layout("The Magic Constant: Euler's Characteristic Formula", [
            "For connected planar graphs, V minus E plus F equals 2.",
            "Let's count the vertices, edges, and faces.",
            "Eight vertices minus twelve edges plus six faces equals two.",
            "Stretching the graph doesn't change this constant.",
            "This invariant is called the Euler characteristic."
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # formula "V - E + F = 2"
        formula = MathTex("V", "-", "E", "+", "F", "=", "2", font_size=48, color="#FFFFFF")
        self.place_in_area(formula, "A1", "A6")
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Cube Schlegel Diagram Vertices (Inner square and Outer square)
        inner_pts = [np.array([x, y, 0]) for x in [-0.5, 0.5] for y in [-0.5, 0.5]]
        # reorder for square loop
        inner_pts = [inner_pts[0], inner_pts[1], inner_pts[3], inner_pts[2]]
        outer_pts = [p * 2.5 for p in inner_pts]
        
        vertices = VGroup(*[Dot(p, color="#00FFFF", radius=0.08) for p in inner_pts + outer_pts])
        
        edges_idx = [
            (0,1), (1,2), (2,3), (3,0), # Inner
            (4,5), (5,6), (6,7), (7,4), # Outer
            (0,4), (1,5), (2,6), (3,7)  # Connections
        ]
        
        edges = VGroup(*[
            Line(vertices[i].get_center(), vertices[j].get_center(), color="#FFFFFF", stroke_width=2)
            for i, j in edges_idx
        ])

        planar_graph = VGroup(edges, vertices)
        # Issue 22: self.place_in_area(planar_graph, 'D3', 'F6', scale_factor=0.75)
        self.place_in_area(planar_graph, "D3", "F6", scale_factor=0.75)
        
        self.play(Create(planar_graph))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Issue 21: labels at C2, C4, C6
        v_label = MathTex("V=8", color="#00FFFF", font_size=32)
        e_label = MathTex("E=12", color="#FFFFFF", font_size=32)
        f_label = MathTex("F=6", color="#FFFF00", font_size=32)
        
        self.place_at_grid(v_label, "C2", scale_factor=0.8)
        self.place_at_grid(e_label, "C4", scale_factor=0.8)
        self.place_at_grid(f_label, "C6", scale_factor=0.8)
        
        self.play(FadeIn(v_label), FadeIn(e_label), FadeIn(f_label))
        self.wait(1)

        # Issue 23: result_formula in area B1-B6
        result_formula = MathTex("8", "-", "12", "+", "6", "=", "2", font_size=48)
        result_formula[0].set_color("#00FFFF")
        result_formula[2].set_color("#FFFFFF")
        result_formula[4].set_color("#FFFF00")
        result_formula[6].set_color("#00FF00")
        
        self.place_in_area(result_formula, "B1", "B6", scale_factor=0.9)
        
        # Move values into the numerical formula
        self.play(
            ReplacementTransform(v_label[0][2:].copy(), result_formula[0]),
            ReplacementTransform(e_label[0][2:].copy(), result_formula[2]),
            ReplacementTransform(f_label[0][2:].copy(), result_formula[4]),
            FadeIn(result_formula[1]), FadeIn(result_formula[3]),
            FadeIn(result_formula[5]), FadeIn(result_formula[6])
        )
        
        # Animate calculation result "2"
        self.play(Indicate(result_formula[6], color="#00FF00", scale_factor=1.5))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Stretching vertices
        # We need to preserve the group structure for updaters
        # Using a ValueTracker to control deformation
        stretch_tracker = ValueTracker(0)
        
        # Generate random directions for vertices to stretch
        # Seed for reproducibility
        np.random.seed(42)
        stretch_dirs = [np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0]) for _ in range(8)]
        
        orig_v_pos = [v.get_center().copy() for v in vertices]
        
        def update_v(m):
            val = stretch_tracker.get_value()
            for i, v in enumerate(m):
                v.move_to(orig_v_pos[i] + val * stretch_dirs[i])
        
        def update_e(m):
            for i, (idx1, idx2) in enumerate(edges_idx):
                m[i].put_start_and_end_on(vertices[idx1].get_center(), vertices[idx2].get_center())

        vertices.add_updater(update_v)
        edges.add_updater(update_e)
        
        self.play(stretch_tracker.animate.set_value(1), run_time=2, rate_func=there_and_back)
        
        vertices.remove_updater(update_v)
        edges.remove_updater(update_e)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Final emphasis on the characteristic
        invariant_text = Text("Euler Characteristic", font_size=24, color=YELLOW)
        self.place_at_grid(invariant_text, "A3", scale_factor=0.8) # Place near the top formula
        
        self.play(Write(invariant_text))
        self.play(Indicate(formula, color=YELLOW), Indicate(result_formula, color=YELLOW))
        self.wait(2)
