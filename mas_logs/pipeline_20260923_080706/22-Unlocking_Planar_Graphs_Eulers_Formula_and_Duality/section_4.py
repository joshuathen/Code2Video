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
            "Original vertices become dual faces.",
            "Original faces become dual vertices.",
            "These structures demonstrate mathematical symmetry.",
            "Original graph edges map to dual edges.",
            "This duality reveals core graph relationships."
        ]
        self.setup_layout("The Duality Connection", lecture_lines)
        
        # Asset Loading
        graph_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        
        # Original graph (simplified representation)
        vertices = VGroup(*[Dot(color=BLUE) for _ in range(4)])
        edges = VGroup(
            Line(np.array([-0.5, 0.5, 0]), np.array([0.5, 0.5, 0]), color=GRAY),
            Line(np.array([0.5, 0.5, 0]), np.array([0.5, -0.5, 0]), color=GRAY),
            Line(np.array([0.5, -0.5, 0]), np.array([-0.5, -0.5, 0]), color=GRAY),
            Line(np.array([-0.5, -0.5, 0]), np.array([-0.5, 0.5, 0]), color=GRAY)
        )
        original_graph = VGroup(edges, vertices)
        self.place_in_area(original_graph, 'B2', 'D4', scale_factor=0.6)
        
        # Dual graph
        dual_vertex = Dot(color=GOLD, radius=0.1)
        self.place_at_grid(dual_vertex, 'C3', scale_factor=0.5)
        dual_vertex.set_opacity(0)
        
        # Add to scene
        self.add(graph_icon, original_graph, dual_vertex)
        self.place_at_grid(graph_icon, 'C3', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GOLD), dual_vertex.animate.set_opacity(1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(GRAY))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
