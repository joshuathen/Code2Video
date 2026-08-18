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
        self.setup_layout("The Concept of Dual Graphs", [
            "Dual graphs transform faces into new vertices.",
            "Connect vertices if their faces share an edge.",
            "Hexagonal honeybee grids form triangular dual lattices."
        ])
        
        # Original graph setup
        nodes = VGroup(*[Dot(color=WHITE) for _ in range(4)])
        graph_group = VGroup(
            nodes[0], nodes[1], nodes[2], nodes[3]
        )
        
        # Improvement 31 & 46
        self.place_in_area(graph_group, 'C4', 'E5', scale_factor=0.9)
        
        edges = VGroup(
            Line(nodes[0].get_center(), nodes[1].get_center(), color="#00FFFF"),
            Line(nodes[1].get_center(), nodes[2].get_center(), color="#00FFFF"),
            Line(nodes[2].get_center(), nodes[3].get_center(), color="#00FFFF"),
            Line(nodes[3].get_center(), nodes[0].get_center(), color="#00FFFF"),
            Line(nodes[0].get_center(), nodes[2].get_center(), color="#00FFFF")
        )
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(nodes), Create(edges))
        self.lecture[0].set_color("#FF0000")
        
        # Asset integration: honeybee icons
        bee1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/honeybee.svg", color="#FF0000")
        bee2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/honeybee.svg", color="#FF0000")
        
        # Improvement 33 & 48
        self.place_at_grid(bee1, 'D5', scale_factor=0.3)
        self.place_at_grid(bee2, 'E5', scale_factor=0.3) # Adjusted for visual separation
        
        dual_nodes = VGroup(bee1, bee2)
        self.play(FadeIn(dual_nodes))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        dual_edge = DashedLine(bee1.get_center(), bee2.get_center(), color="#FFFF00")
        self.play(Create(dual_edge))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
