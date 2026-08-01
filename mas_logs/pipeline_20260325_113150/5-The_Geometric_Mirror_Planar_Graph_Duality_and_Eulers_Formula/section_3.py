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
        # Mandatory layout setup
        lecture_lines = [
            "Start with a planar graph containing two interior faces.",
            "Place a dual vertex inside every face, including the exterior.",
            "Connect dual vertices by crossing an edge of the original.",
            "Continue until every original edge has a corresponding dual edge.",
            "The resulting structure is the dual of the original graph."
        ]
        self.setup_layout("The Dual Transformation: Building the Mirror Graph", lecture_lines)

        # Colors
        GRAY_COLOR = "#808080"
        RED_COLOR = "#FF0000"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(GRAY_COLOR)
        
        # Define vertices for a square with a diagonal
        # We use relative points initially
        v_pts = [np.array([-1, 1, 0]), np.array([1, 1, 0]), np.array([1, -1, 0]), np.array([-1, -1, 0])]
        v_dots = VGroup(*[Dot(p, color=GRAY_COLOR) for p in v_pts])
        e_lines = VGroup(
            Line(v_pts[0], v_pts[1], color=GRAY_COLOR), # Top
            Line(v_pts[1], v_pts[2], color=GRAY_COLOR), # Right
            Line(v_pts[2], v_pts[3], color=GRAY_COLOR), # Bottom
            Line(v_pts[3], v_pts[0], color=GRAY_COLOR), # Left
            Line(v_pts[0], v_pts[2], color=GRAY_COLOR)  # Diagonal
        )
        planar_graph = VGroup(v_dots, e_lines)
        
        # ISSUE 27 FIX: Frame the original graph
        self.place_in_area(planar_graph, 'B2', 'E5', scale_factor=1.2)
        
        self.play(Create(planar_graph))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RED_COLOR)
        
        # Calculate face centers from the transformed planar_graph vertices
        # Face 1: Vertices 0, 1, 2 (Top-Right triangle)
        # Face 2: Vertices 0, 2, 3 (Bottom-Left triangle)
        p0 = v_dots[0].get_center()
        p1 = v_dots[1].get_center()
        p2 = v_dots[2].get_center()
        p3 = v_dots[3].get_center()
        
        dv1 = Dot((p0 + p1 + p2)/3, color=RED_COLOR, radius=0.12)
        dv2 = Dot((p0 + p2 + p3)/3, color=RED_COLOR, radius=0.12)
        
        self.play(FadeIn(dv1), FadeIn(dv2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED_COLOR)
        
        # Dual vertex for the infinite outer face
        exterior_dual_vertex = Dot(color=RED_COLOR, radius=0.12)
        
        # ISSUE 28 FIX: Position the outer vertex at C6
        self.place_at_grid(exterior_dual_vertex, 'C6', scale_factor=0.8)
        
        self.play(FadeIn(exterior_dual_vertex))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED_COLOR)
        
        # Dual edge crossing the original diagonal (p0-p2)
        de_diagonal = DashedLine(dv1.get_center(), dv2.get_center(), color=RED_COLOR)
        self.play(Create(de_diagonal))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(RED_COLOR)
        
        p_dv1 = dv1.get_center()
        p_dv2 = dv2.get_center()
        p_out = exterior_dual_vertex.get_center()
        
        # Create dual edges crossing boundary edges (Top, Right, Bottom, Left)
        # Using arcs/curves to ensure they cross only the intended original edge
        de_top = DashedVMobject(ArcBetweenPoints(p_dv1, p_out, angle=-PI/2), num_dashes=20).set_color(RED_COLOR)
        de_right = DashedVMobject(ArcBetweenPoints(p_dv1, p_out, angle=0.2), num_dashes=15).set_color(RED_COLOR)
        de_bottom = DashedVMobject(ArcBetweenPoints(p_dv2, p_out, angle=PI/2), num_dashes=20).set_color(RED_COLOR)
        de_left = DashedVMobject(ArcBetweenPoints(p_dv2, p_out, angle=-1.4*PI), num_dashes=35).set_color(RED_COLOR)
        
        dual_edges_group = VGroup(de_top, de_right, de_bottom, de_left)
        self.play(Create(dual_edges_group))
        self.wait(1)
        
        # Define the complete dual graph group for final framing
        dual_graph_group = VGroup(dv1, dv2, exterior_dual_vertex, de_diagonal, dual_edges_group)
        
        # ISSUE 29 FIX: Re-frame the dual graph structure
        self.place_in_area(dual_graph_group, 'B2', 'F6', scale_factor=1.0)
        
        # Fade out original graph and leave the dual visible
        self.play(FadeOut(planar_graph))
        self.wait(2)
