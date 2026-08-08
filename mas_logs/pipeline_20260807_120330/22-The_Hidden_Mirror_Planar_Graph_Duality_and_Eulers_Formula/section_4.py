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
        lecture_lines = [
            "G and G-star share a deep mathematical symmetry.",
            "Vertices in G-star equal the faces in G.",
            "Faces in G-star equal the vertices in G.",
            "The number of edges remains exactly the same.",
            "Euler's formula holds perfectly for both reflections."
        ]
        self.setup_layout("The Symmetry of Elements", lecture_lines)

        # Colors
        COLOR_G = "#FF8C00"
        COLOR_GSTAR = "#00FFFF"
        COLOR_HIGHLIGHT = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))

        # Graph G: Square (Planar Projection)
        v_g = VGroup(
            Dot(color=COLOR_G), Dot(color=COLOR_G),
            Dot(color=COLOR_G), Dot(color=COLOR_G)
        ).arrange_in_grid(2, 2, buff=1.0)
        
        e_g = VGroup(
            Line(v_g[0].get_center(), v_g[1].get_center(), color=COLOR_G),
            Line(v_g[1].get_center(), v_g[3].get_center(), color=COLOR_G),
            Line(v_g[3].get_center(), v_g[2].get_center(), color=COLOR_G),
            Line(v_g[2].get_center(), v_g[0].get_center(), color=COLOR_G),
        )
        graph_g = VGroup(v_g, e_g)
        # Fix: Issue 30 - Shift graph down to B1-D3
        self.place_in_area(graph_g, "B1", "D3", scale_factor=0.6)
        
        label_g = Text("Graph G", color=COLOR_G, font_size=24)
        self.place_at_grid(label_g, "A2", scale_factor=0.8)

        # Graph G*: Dual of Square (2 vertices, 4 edges between them)
        v_gstar = VGroup(
            Dot(color=COLOR_GSTAR),
            Dot(color=COLOR_GSTAR)
        ).arrange(RIGHT, buff=1.5)
        
        # 4 multi-edges between the two dual vertices
        e_gstar = VGroup(
            ArcBetweenPoints(v_gstar[0].get_center(), v_gstar[1].get_center(), angle=PI/2, color=COLOR_GSTAR),
            ArcBetweenPoints(v_gstar[0].get_center(), v_gstar[1].get_center(), angle=-PI/2, color=COLOR_GSTAR),
            ArcBetweenPoints(v_gstar[0].get_center(), v_gstar[1].get_center(), angle=PI, color=COLOR_GSTAR),
            ArcBetweenPoints(v_gstar[0].get_center(), v_gstar[1].get_center(), angle=-PI, color=COLOR_GSTAR),
        )
        graph_gstar = VGroup(v_gstar, e_gstar)
        # Fix: Issue 30 - Shift graph down to B4-D6
        self.place_in_area(graph_gstar, "B4", "D6", scale_factor=0.6)
        
        label_gstar = Text("Dual G*", color=COLOR_GSTAR, font_size=24)
        self.place_at_grid(label_gstar, "A5", scale_factor=0.8)

        self.play(Create(graph_g), Write(label_g))
        self.play(Create(graph_gstar), Write(label_gstar))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Faces of G: 2 (Internal square area, External background area)
        face_g1 = Square(side_length=0.7, fill_opacity=0.4, color=COLOR_G, stroke_width=0).move_to(graph_g.get_center())
        face_g2 = Annulus(inner_radius=0.6, outer_radius=1.0, fill_opacity=0.2, color=COLOR_G, stroke_width=0).move_to(graph_g.get_center())
        
        self.play(FadeIn(face_g1), FadeIn(face_g2))
        self.play(Indicate(v_gstar, color=COLOR_HIGHLIGHT, scale_factor=1.5))
        self.play(FadeOut(face_g1), FadeOut(face_g2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Vertices of G -> Faces of G*
        self.play(Indicate(v_g, color=COLOR_HIGHLIGHT, scale_factor=1.5))
        
        # Representing faces of G* (regions between arcs)
        f_gstar_highs = VGroup(*[
            Dot(graph_gstar.get_center() + UP*0.5, radius=0.3, color=COLOR_GSTAR, fill_opacity=0.5),
            Dot(graph_gstar.get_center() + DOWN*0.5, radius=0.3, color=COLOR_GSTAR, fill_opacity=0.5),
            Dot(graph_gstar.get_center() + UP*0.2, radius=0.1, color=COLOR_GSTAR, fill_opacity=0.5),
            Dot(graph_gstar.get_center() + DOWN*0.2, radius=0.1, color=COLOR_GSTAR, fill_opacity=0.5),
        ])
        
        self.play(FadeIn(f_gstar_highs))
        self.wait(1)
        self.play(FadeOut(f_gstar_highs))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Edges comparison: Highlight 1-to-1 correspondence
        for i in range(4):
            self.play(
                Indicate(e_g[i], color=COLOR_HIGHLIGHT),
                Indicate(e_gstar[i], color=COLOR_HIGHLIGHT),
                run_time=0.4
            )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Show Euler's Formula for both
        f_g_text = MathTex("V - E + F = 2", color=WHITE)
        f_g_vals = MathTex("4 - 4 + 2 = 2", color=COLOR_G)
        # Fix: Issue 31 - Shift formula text down to row E
        self.place_in_area(f_g_text, "E1", "E3", scale_factor=0.8)
        self.place_in_area(f_g_vals, "F1", "F3", scale_factor=0.8)

        f_gs_text = MathTex("V^* - E^* + F^* = 2", color=WHITE)
        f_gs_vals = MathTex("2 - 4 + 4 = 2", color=COLOR_GSTAR)
        # Fix: Issue 31 - Shift formula text down to row E
        self.place_in_area(f_gs_text, "E4", "E6", scale_factor=0.8)
        self.place_in_area(f_gs_vals, "F4", "F6", scale_factor=0.8)

        self.play(Write(f_g_text), Write(f_gs_text))
        self.play(Write(f_g_vals), Write(f_gs_vals))
        
        # Final highlight showing the swapping of V and F
        rect_v_g = SurroundingRectangle(f_g_vals[0][0], color=COLOR_HIGHLIGHT) # V=4
        rect_f_gs = SurroundingRectangle(f_gs_vals[0][4], color=COLOR_HIGHLIGHT) # F*=4
        
        rect_f_g = SurroundingRectangle(f_g_vals[0][4], color=COLOR_HIGHLIGHT) # F=2
        rect_v_gs = SurroundingRectangle(f_gs_vals[0][0], color=COLOR_HIGHLIGHT) # V*=2
        
        self.play(Create(rect_v_g), Create(rect_f_gs))
        self.wait(0.5)
        self.play(
            ReplacementTransform(rect_v_g, rect_f_g), 
            ReplacementTransform(rect_f_gs, rect_v_gs)
        )
        self.wait(2)
