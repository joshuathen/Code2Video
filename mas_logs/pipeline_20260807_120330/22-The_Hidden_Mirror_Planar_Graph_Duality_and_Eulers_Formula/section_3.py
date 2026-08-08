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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup layout
        self.setup_layout("The Transformation: Constructing the Dual Graph", 
                          ["Every face in G becomes a dual vertex.", 
                           "Dual vertices connect across shared edge boundaries.", 
                           "This process builds the dual graph G-star."])
        
        # Colors
        COLOR_G = "#FF8C00"
        COLOR_G_STAR = "#00FFFF"
        COLOR_HIGHLIGHT = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Highlight the first lecture line in #FFFF00. 
        # Take a planar graph in #FF8C00 and place a #00FFFF dot (dual vertex) in every face, including the exterior.
        
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))

        # Original Graph G (Triangle for clarity)
        # Positions chosen to be central and distinct
        v1_pos = self.grid["B2"]
        v2_pos = self.grid["B5"]
        v3_pos = self.grid["F3"]
        
        v1 = Dot(v1_pos, color=COLOR_G)
        v2 = Dot(v2_pos, color=COLOR_G)
        v3 = Dot(v3_pos, color=COLOR_G)
        g_vertices = VGroup(v1, v2, v3)
        
        e1 = Line(v1_pos, v2_pos, color=COLOR_G)
        e2 = Line(v2_pos, v3_pos, color=COLOR_G)
        e3 = Line(v3_pos, v1_pos, color=COLOR_G)
        g_edges = VGroup(e1, e2, e3)
        
        g_graph = VGroup(g_edges, g_vertices)
        g_label = MathTex("G", color=COLOR_G)
        # Fix: Issue #25 - Move g_label from F6 to E5
        self.place_at_grid(g_label, "E5", scale_factor=1.0)

        self.play(Create(g_graph), Write(g_label))
        self.wait(1)

        # Dual Vertices
        # Inner face vertex - center of triangle
        dv_inner = Dot(self.grid["C3"], color=COLOR_G_STAR)
        # Outer face vertex - far enough to wrap around
        dv_outer = Dot(self.grid["A6"], color=COLOR_G_STAR)
        dual_vertices = VGroup(dv_inner, dv_outer)

        self.play(FadeIn(dual_vertices))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight the second lecture line in #FFFF00. 
        # Draw #00FFFF dashed lines (dual edges) crossing each edge of the #FF8C00 graph to connect the dots.
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HIGHLIGHT)
        )

        # Dual Edges (Solid paths for transformation)
        # 1. Crossing e1 (Top edge)
        de1_path = ArcBetweenPoints(dv_inner.get_center(), dv_outer.get_center(), angle=-0.5, color=COLOR_G_STAR)
        # 2. Crossing e2 (Right edge)
        de2_path = ArcBetweenPoints(dv_inner.get_center(), dv_outer.get_center(), angle=0.8, color=COLOR_G_STAR)
        # 3. Crossing e3 (Left edge)
        de3_path = ArcBetweenPoints(dv_inner.get_center(), dv_outer.get_center(), angle=-2.5, color=COLOR_G_STAR)
        
        dual_edges_solid = VGroup(de1_path, de2_path, de3_path)
        
        # Dashed versions for the step-by-step construction
        # Note: Using DashedVMobject outside always_redraw per instructions
        dual_edges_dashed = VGroup(*[DashedVMobject(edge, num_dashes=15) for edge in dual_edges_solid])

        self.play(Create(dual_edges_dashed, run_time=3))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight the third lecture line in #FFFF00. 
        # Fade out the #FF8C00 graph, leaving only the #00FFFF dots and dashed lines, 
        # then solidify the dashed lines.
        
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )

        g_star_label = MathTex("G^*", color=COLOR_G_STAR)
        # Fix: Issue #26 - Move g_star_label from A4 to B4
        self.place_at_grid(g_star_label, "B4", scale_factor=1.0)

        self.play(
            FadeOut(g_graph), 
            FadeOut(g_label),
            Write(g_star_label)
        )
        self.wait(1)

        # Solidify the dual edges to complete G*
        self.play(ReplacementTransform(dual_edges_dashed, dual_edges_solid))
        self.wait(2)

        # Final cleanup - return highlights to normal
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
