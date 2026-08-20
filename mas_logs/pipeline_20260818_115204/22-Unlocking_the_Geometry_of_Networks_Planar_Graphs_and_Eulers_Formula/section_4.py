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
        self.setup_layout("The Concept of Dual Graphs", [
            "Dual graphs are created from faces.",
            "Place a vertex in each face.",
            "Connect these vertices across edges."
        ])
        
        # Create a simple hexagon-based graph
        hex_points_rel = [
            np.array([0, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0.866, 0.5, 0]),
            np.array([0.866, -0.5, 0]),
            np.array([0, -1, 0]),
            np.array([-0.866, -0.5, 0]),
            np.array([-0.866, 0.5, 0]),
        ]
        
        graph_edges = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)
        ]
        
        points = VGroup(*[Dot(self.grid["C4"] + p, color=WHITE) for p in hex_points_rel])
        edges = VGroup(*[Line(points[u].get_center(), points[v].get_center(), color=WHITE) for u, v in graph_edges])
        G_total = VGroup(edges, points)
        
        # === Animation for Lecture Line 1 ===
        # Apply layout requirement: self.place_at_grid(graph_vertices, 'B5', scale_factor=0.75)
        self.place_at_grid(G_total, "C4", scale_factor=0.75) 
        self.play(FadeIn(G_total))
        self.lecture[0].set_color(BLUE)
        
        # === Animation for Lecture Line 2 ===
        # Dual vertices - centers of the triangles
        dual_points = []
        for i in range(1, 7):
            next_i = (i % 6) + 1
            tri_center = (points[0].get_center() + points[i].get_center() + points[next_i].get_center()) / 3
            dual_points.append(Dot(tri_center, color="#FF0000"))
        
        dual_G = VGroup(*dual_points)
        # Apply layout requirement: self.place_at_grid(dual_graph_shape, 'B5', scale_factor=0.9)
        # Note: dual_G is already positioned correctly relative to G_total at C4
        self.play(FadeIn(dual_G))
        self.lecture[1].set_color("#FF0000")
        
        # === Animation for Lecture Line 3 ===
        dual_edges = VGroup()
        for i in range(6):
            next_i = (i + 1) % 6
            dual_edges.add(Line(dual_points[i].get_center(), dual_points[next_i].get_center(), color="#FF0000"))
            
        self.play(Create(dual_edges))
        self.lecture[2].set_color("#FF0000")
        self.wait(2)
