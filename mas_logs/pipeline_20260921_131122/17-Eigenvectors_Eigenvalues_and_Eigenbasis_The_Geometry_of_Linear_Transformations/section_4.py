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
        self.setup_layout("Real-World Application: The 'Search Engine' Perspective", 
                          ["Eigenvectors simplify massive data sets effectively.", 
                           "Principal eigenvalues highlight major data trends.", 
                           "PageRank ranks influence using principal eigenvectors."])
        
        # Asset Loading
        server = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg")
        self.place_at_grid(server, "C4", scale_factor=0.3)
        
        # Nodes for web graph
        nodes = VGroup(*[Circle(radius=0.15, color=WHITE, fill_opacity=0.5) for _ in range(6)])
        node_positions = ["A2", "A5", "C1", "C6", "E2", "E5"]
        for node, pos in zip(nodes, node_positions):
            self.place_at_grid(node, pos)
        
        # Connections
        edges = VGroup(
            Line(nodes[0].get_center(), nodes[1].get_center(), stroke_width=2),
            Line(nodes[0].get_center(), nodes[2].get_center(), stroke_width=2),
            Line(nodes[1].get_center(), nodes[3].get_center(), stroke_width=2),
            Line(nodes[2].get_center(), nodes[4].get_center(), stroke_width=2),
            Line(nodes[3].get_center(), nodes[5].get_center(), stroke_width=2),
            Line(nodes[4].get_center(), nodes[5].get_center(), stroke_width=2),
        )

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(server), FadeIn(nodes), Create(edges))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        flow = VGroup(*[Dot(node.get_center(), color="#00FF00") for node in nodes])
        self.play(FadeIn(flow), flow.animate.move_to(server.get_center()))
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        highlight = nodes[5].copy().set_color("#FF00FF").set_stroke(width=4)
        label = Text("PageRank", font_size=20, color="#FF00FF")
        self.place_at_grid(label, "E5")
        self.play(Create(highlight), Write(label))
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.wait(2)
