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
        self.setup_layout("Prerequisite Visualized: Nodes and Paths", [
            "Connectivity defines structure in topology.",
            "Nodes connect to form paths.",
            "Simplified graphs reveal essential connections."
        ])
        
        # Grid Title for constraint
        grid_title = Text("Node Map", font_size=20, color=BLUE)
        self.place_in_area(grid_title, 'A4', 'A6', scale_factor=0.8)
        self.add(grid_title)
        
        # Definition Text for constraint
        definition_text = Text("Vertex: Point\nEdge: Connection", font_size=18, color=GRAY)
        self.place_in_area(definition_text, 'B1', 'D3', scale_factor=0.7)
        self.add(definition_text)
        
        # Define nodes and edges
        node_pos = ["C2", "C5", "E2", "E5"]
        nodes = VGroup(*[Circle(radius=0.2, color=BLUE, fill_opacity=1) for _ in node_pos])
        for i, pos in enumerate(node_pos):
            self.place_at_grid(nodes[i], pos)
        
        edges = VGroup(
            Line(nodes[0].get_center(), nodes[1].get_center(), color=WHITE),
            Line(nodes[1].get_center(), nodes[3].get_center(), color=WHITE),
            Line(nodes[3].get_center(), nodes[2].get_center(), color=WHITE),
            Line(nodes[2].get_center(), nodes[0].get_center(), color=WHITE)
        )
        
        # Place graph group as requested
        graph_nodes = VGroup(nodes, edges)
        self.place_at_grid(graph_nodes, 'C4', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(nodes), Create(edges))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        path = VGroup(edges[0], edges[1])
        self.play(path.animate.set_color(GREEN))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Contract the path
        self.play(
            nodes[0].animate.move_to(nodes[1].get_center()),
            nodes[2].animate.move_to(nodes[3].get_center()),
            path.animate.move_to(nodes[1].get_center())
        )
        self.play(FadeOut(path), nodes[0].animate.move_to(nodes[3].get_center()))
        self.wait(1)
