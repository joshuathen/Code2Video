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
        self.setup_layout("Mathematical Symmetry & Euler's Swap", 
                          ["Dual vertices correspond to the original graph's faces.", 
                           "Dual faces correspond to the original graph's vertices.", 
                           "Symmetry preserves Euler's formula in the dual graph."])

        # Colors
        V_COLOR = BLUE_B
        E_COLOR = ORANGE
        F_COLOR = GREEN_B
        HIGHLIGHT_COLOR = YELLOW

        # === Graph Definitions ===
        
        # Original House Graph G (V=5, E=6, F=3)
        g_center = self.grid["B2"]
        v_coords = [
            g_center + np.array([-0.4, -0.5, 0]), # 0
            g_center + np.array([0.4, -0.5, 0]),  # 1
            g_center + np.array([0.4, 0.3, 0]),   # 2
            g_center + np.array([-0.4, 0.3, 0]),  # 3
            g_center + np.array([0, 0.8, 0])      # 4
        ]
        g_vertices = VGroup(*[Dot(p, color=V_COLOR, radius=0.08) for p in v_coords])
        g_edges_indices = [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (3, 4)]
        g_edges = VGroup(*[Line(v_coords[i], v_coords[j], color=E_COLOR, stroke_width=3) for i, j in g_edges_indices])
        
        # Faces of G (visual representation)
        face1_square = Polygon(v_coords[0], v_coords[1], v_coords[2], v_coords[3], color=F_COLOR, fill_opacity=0.3, stroke_width=0)
        face2_tri = Polygon(v_coords[2], v_coords[3], v_coords[4], color=F_COLOR, fill_opacity=0.3, stroke_width=0)
        g_faces = VGroup(face1_square, face2_tri)
        
        g_full = VGroup(g_faces, g_edges, g_vertices)
        
        # Dual Graph G* (V*=3, E*=6, F*=5)
        gs_center = self.grid["B5"]
        vs_coords = [
            gs_center + np.array([0, -0.1, 0]),   # 0 (center of square)
            gs_center + np.array([0, 0.55, 0]),   # 1 (center of triangle)
            gs_center + np.array([0.8, 0, 0])     # 2 (outside)
        ]
        gs_vertices = VGroup(*[Dot(p, color=V_COLOR, radius=0.08) for p in vs_coords])
        
        # Dual edges: 1 between V*0-V*1, 3 between V*0-V*2, 2 between V*1-V*2
        gs_edges = VGroup(
            Line(vs_coords[0], vs_coords[1], color=E_COLOR, stroke_width=3),
            ArcBetweenPoints(vs_coords[0], vs_coords[2], angle=TAU/6, color=E_COLOR, stroke_width=3),
            ArcBetweenPoints(vs_coords[0], vs_coords[2], angle=-TAU/6, color=E_COLOR, stroke_width=3),
            ArcBetweenPoints(vs_coords[0], vs_coords[2], angle=TAU/3, color=E_COLOR, stroke_width=3),
            ArcBetweenPoints(vs_coords[1], vs_coords[2], angle=TAU/6, color=E_COLOR, stroke_width=3),
            ArcBetweenPoints(vs_coords[1], vs_coords[2], angle=-TAU/6, color=E_COLOR, stroke_width=3)
        )
        gs_full = VGroup(gs_edges, gs_vertices)

        # Labels
        label_g = Text("Graph G", font_size=20, color=WHITE).next_to(g_vertices, DOWN, buff=0.3)
        label_gs = Text("Dual G*", font_size=20, color=WHITE).next_to(gs_vertices, DOWN, buff=0.3)

        # === Animation for Lecture Line 1 ===
        # "Dual vertices correspond to the original graph's faces."
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        self.play(Create(g_edges), Create(g_vertices), Write(label_g))
        self.wait(0.5)
        self.play(Create(gs_edges), Create(gs_vertices), Write(label_gs))
        self.wait(0.5)
        
        # Highlight G's faces and G*'s vertices
        self.play(FadeIn(g_faces), gs_vertices.animate.scale(1.5).set_color(HIGHLIGHT_COLOR))
        self.play(gs_vertices.animate.scale(1/1.5).set_color(V_COLOR))
        
        val_f = MathTex("F = 3", font_size=24, color=F_COLOR)
        val_vs = MathTex("V^* = 3", font_size=24, color=V_COLOR)
        self.place_at_grid(val_f, "D2")
        self.place_at_grid(val_vs, "D4") # Resolved Issue 24: Moved from D5 to D4
        
        arrow1 = Arrow(val_f.get_right(), val_vs.get_left(), color=WHITE, buff=0.1)
        
        self.play(Write(val_f), Write(val_vs))
        self.play(GrowArrow(arrow1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Dual faces correspond to the original graph's vertices."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Highlight G's vertices and suggest G*'s faces
        self.play(Indicate(g_vertices, color=HIGHLIGHT_COLOR), Indicate(gs_edges, color=HIGHLIGHT_COLOR))
        
        val_v = MathTex("V = 5", font_size=24, color=V_COLOR)
        val_fs = MathTex("F^* = 5", font_size=24, color=F_COLOR)
        self.place_at_grid(val_v, "E2")
        self.place_at_grid(val_fs, "E4") # Resolved Issue 24: Moved from E5 to E4
        
        arrow2 = Arrow(val_v.get_right(), val_fs.get_left(), color=WHITE, buff=0.1)
        
        self.play(Write(val_v), Write(val_fs))
        self.play(GrowArrow(arrow2))
        
        # Flash edges
        self.play(Flash(g_edges, color=E_COLOR, line_length=0.2), Flash(gs_edges, color=E_COLOR, line_length=0.2))
        
        val_e = MathTex("E = 6", font_size=20, color=E_COLOR)
        val_es = MathTex("E^* = 6", font_size=20, color=E_COLOR)
        self.place_at_grid(val_e, "C5") # Resolved Issue 25: Moved from D3 to C5
        self.place_at_grid(val_es, "C2") # Resolved Issue 25: Moved from D4 to C2
        self.play(Write(val_e), Write(val_es))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Symmetry preserves Euler's formula in the dual graph."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        formula = MathTex("V^*", "-", "E^*", "+", "F^*", "=", "2", font_size=28)
        formula_nums = MathTex("3", "-", "6", "+", "5", "=", "2", font_size=28)
        self.place_in_area(formula, "F2", "F4") # Resolved Issue 26: Reduced area from F2-F5 to F2-F4
        self.place_in_area(formula_nums, "F2", "F4") # Resolved Issue 26: Reduced area
        
        formula[0].set_color(V_COLOR)
        formula[2].set_color(E_COLOR)
        formula[4].set_color(F_COLOR)
        
        formula_nums[0].set_color(V_COLOR)
        formula_nums[2].set_color(E_COLOR)
        formula_nums[4].set_color(F_COLOR)

        self.play(Write(formula))
        self.wait(1)
        self.play(Transform(formula, formula_nums))
        self.wait(1)
        
        # Symmetry swap
        # We must swap all objects associated with G and G*.
        # G items: g_full, label_g, val_v (E2), val_f (D2), val_e (C5)
        # G* items: gs_full, label_gs, val_vs (D4), val_fs (E4), val_es (C2)
        # Shift amounts: 
        # Left-positioned (Col 2) move +3 to Col 5.
        # Right-positioned (Col 5) move -3 to Col 2.
        # Col 4 move -2 to Col 2? No, let's keep it simple: swap columns 2 and 5, and adjust col 4.
        
        self.play(
            # G things at Col 2 move to Col 5 (+3 units)
            VGroup(g_full, label_g, val_v, val_f).animate.shift(RIGHT * 3),
            # G* things at Col 5 move to Col 2 (-3 units)
            VGroup(gs_full, label_gs, val_e).animate.shift(LEFT * 3),
            # Crossed items: val_es at C2 moves to C5 (+3), val_vs/val_fs at Col 4 move to Col 2 (-2)
            val_es.animate.shift(RIGHT * 3),
            VGroup(val_vs, val_fs).animate.shift(LEFT * 2),
            # Arrow handling
            arrow1.animate.shift(LEFT * 1).scale(0.5), # Just fading them out or shrinking is better
            arrow2.animate.shift(LEFT * 1).scale(0.5),
            run_time=2
        )
        self.wait(2)
