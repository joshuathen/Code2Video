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
        # Initial Setup
        lecture_lines = [
            "Place a new vertex inside every original face.",
            "Remember to place a vertex in the outer region.",
            "Connect these vertices by crossing every original edge once.",
            "This cube’s dual construction reveals a symmetric octahedron.",
            "Duality exposes the deep links between Platonic solids."
        ]
        self.setup_layout("Step-by-Step Construction of a Dual Graph", lecture_lines)

        # Base Cube Graph construction (Schlegel diagram)
        # Defining vertices relative to origin for easier area placement
        vi = [np.array([x, y, 0]) for x, y in [(-0.6, 0.6), (0.6, 0.6), (0.6, -0.6), (-0.6, -0.6)]]
        vo = [np.array([x, y, 0]) for x, y in [(-1.5, 1.5), (1.5, 1.5), (1.5, -1.5), (-1.5, -1.5)]]
        
        cube_graph = VGroup()
        for i in range(4):
            cube_graph.add(Line(vi[i], vi[(i+1)%4], color=WHITE)) # Inner square
            cube_graph.add(Line(vo[i], vo[(i+1)%4], color=WHITE)) # Outer square
            cube_graph.add(Line(vi[i], vo[i], color=WHITE))       # Connecting edges
            
        # [Issue 38] Resolve small/poorly anchored cube by using grid area
        self.place_in_area(cube_graph, 'B2', 'E5', scale_factor=1.1)
        
        # Capture precise vertex positions after placement for dual vertex calculation
        # Order: 0:InnerTop, 1:OuterTop, 2:TopLeftConn, 3:InnerRight, 4:OuterRight, 5:TopRightConn, etc.
        pi0 = cube_graph[0].get_start()  # Inner Top-Left
        pi1 = cube_graph[0].get_end()    # Inner Top-Right
        pi2 = cube_graph[3].get_end()    # Inner Bottom-Right
        pi3 = cube_graph[6].get_end()    # Inner Bottom-Left
        po0 = cube_graph[1].get_start()  # Outer Top-Left
        po1 = cube_graph[1].get_end()    # Outer Top-Right
        po2 = cube_graph[4].get_end()    # Outer Bottom-Right
        po3 = cube_graph[7].get_end()    # Outer Bottom-Left

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF0000")
        self.play(Create(cube_graph))

        # Face Centers (Internal)
        p_inner = (pi0 + pi1 + pi2 + pi3) / 4
        p_top = (pi0 + pi1 + po1 + po0) / 4
        p_right = (pi1 + pi2 + po2 + po1) / 4
        p_bottom = (pi2 + pi3 + po3 + po2) / 4
        p_left = (pi3 + pi0 + po0 + po3) / 4
        
        dots_internal = VGroup(*[Dot(p, color="#FF0000", radius=0.08) for p in [p_inner, p_top, p_right, p_bottom, p_left]])
        self.play(FadeIn(dots_internal, shift=UP))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        # [Issue 39] Fix: Position outer vertex at A4 to avoid title collision
        outer_vertex = Dot(color="#FF0000", radius=0.08)
        self.place_at_grid(outer_vertex, 'A4', scale_factor=0.8)
        self.play(FadeIn(outer_vertex, scale=1.5))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        p_out = outer_vertex.get_center()
        
        # Dual edges crossing cube edges
        dual_edges = VGroup(
            Line(p_inner, p_top, color="#FF0000", stroke_width=4),
            Line(p_inner, p_right, color="#FF0000", stroke_width=4),
            Line(p_inner, p_bottom, color="#FF0000", stroke_width=4),
            Line(p_inner, p_left, color="#FF0000", stroke_width=4),
            Line(p_out, p_top, color="#FF0000", stroke_width=4),
            Line(p_out, p_right, color="#FF0000", stroke_width=4),
            Line(p_out, p_bottom, color="#FF0000", stroke_width=4),
            Line(p_out, p_left, color="#FF0000", stroke_width=4),
            Line(p_top, p_right, color="#FF0000", stroke_width=4),
            Line(p_right, p_bottom, color="#FF0000", stroke_width=4),
            Line(p_bottom, p_left, color="#FF0000", stroke_width=4),
            Line(p_left, p_top, color="#FF0000", stroke_width=4)
        )
        self.play(Create(dual_edges))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF0000")
        
        # Define Symmetric Octahedron
        ov = [
            np.array([0, 1.2, 0]), np.array([0, -1.2, 0]), # Vertices (Top/Bottom)
            np.array([-1, 0, 0]), np.array([1, 0, 0]),    # Vertices (Left/Right)
            np.array([-0.4, 0.4, 0]), np.array([0.4, -0.4, 0]) # Vertices (Depth-simulated)
        ]
        octa_lines = VGroup()
        connections = [(0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5), (2,4), (4,3), (3,5), (5,2)]
        for i, j in connections:
            octa_lines.add(Line(ov[i], ov[j], color="#FF0000", stroke_width=6))
        
        octa_dots = VGroup(*[Dot(p, color="#FF0000", radius=0.08) for p in ov])
        octahedron = VGroup(octa_dots, octa_lines)
        
        # [Issue 40] Fix: Centrally align the octahedron within the grid area
        self.place_in_area(octahedron, 'B2', 'E5', scale_factor=1.2)
        
        # Group current dual elements for transformation
        current_dual = VGroup(dots_internal, outer_vertex, dual_edges)
        
        self.play(
            FadeOut(cube_graph),
            Transform(current_dual, octahedron),
            run_time=2
        )

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF0000")
        self.play(Circumscribe(current_dual, color="#FF0000"))
        self.wait(2)
