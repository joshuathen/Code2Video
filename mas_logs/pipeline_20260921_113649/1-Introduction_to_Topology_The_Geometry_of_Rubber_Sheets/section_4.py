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
        self.setup_layout("Real-World Application: The Seven Bridges of Königsberg", 
                          ["Euler solved the Seven Bridges of Konigsberg problem.", 
                           "He reduced the city to a graph diagram.", 
                           "This birthed the field of graph theory."])
        
        # Load Assets
        river = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/river.svg")
        bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        
        # Create Graph for Konigsberg
        nodes = {1: [0, 1, 0], 2: [2, 1, 0], 3: [1, 0, 0], 4: [1, 2, 0]}
        edges = [(1, 4), (1, 4), (1, 3), (1, 3), (2, 4), (2, 4), (2, 3)]
        graph = Graph(nodes.keys(), edges, layout=nodes, labels=True)
        graph_label = Text("Königsberg Graph", font_size=20)
        
        self.place_in_area(graph, "B3", "E5", scale_factor=0.75)
        self.place_at_grid(graph_label, "D2", scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(river), Create(graph))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight bridges
        for edge in graph.edges:
             graph.edges[edge].add(bridge.copy().scale(0.5).move_to(graph.edges[edge].get_center()))
        self.play(graph.animate.set_color(RED), run_time=2)
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Trace path (simplistic highlighting)
        path = VGroup(*[graph.edges[edge] for edge in edges])
        self.play(Indicate(path), run_time=2)
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
