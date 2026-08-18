from manim import *
import numpy as np

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
        self.setup_layout("Practical Application: The Map Coloring Challenge", 
                          ["Euler's formula helps solve complex spatial constraints.", 
                           "Dual graphs simplify map coloring challenges.", 
                           "Model regions as graphs for efficient tower placement."])
        
        # Colors for graph nodes
        colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"]

        # === Animation for Lecture Line 1 ===
        # Show a map (simplified as 4 regions)
        rect1 = Rectangle(height=1.5, width=1.5, fill_opacity=0.5, color=colors[0])
        rect2 = Rectangle(height=1.5, width=1.5, fill_opacity=0.5, color=colors[1])
        rect3 = Rectangle(height=1.5, width=1.5, fill_opacity=0.5, color=colors[2])
        rect4 = Rectangle(height=1.5, width=1.5, fill_opacity=0.5, color=colors[3])
        map_group = VGroup(rect1, rect2, rect3, rect4).arrange_in_grid(buff=0.1, rows=2, cols=2)
        self.place_in_area(map_group, "A1", "C6", scale_factor=0.6)
        self.play(FadeIn(map_group))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Assign each region a vertex and draw edges
        nodes = [Dot(color=WHITE).move_to(r.get_center()) for r in map_group]
        edges = [Line(nodes[0].get_center(), nodes[1].get_center(), color=WHITE),
                 Line(nodes[1].get_center(), nodes[3].get_center(), color=WHITE),
                 Line(nodes[3].get_center(), nodes[2].get_center(), color=WHITE),
                 Line(nodes[2].get_center(), nodes[0].get_center(), color=WHITE)]
        graph = VGroup(*nodes, *edges)
        self.add(graph)
        self.play(Create(graph))
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Coloring the graph vertices using 4 colors
        new_nodes = VGroup(*[Dot(color=colors[i], radius=0.15).move_to(nodes[i].get_center()) for i in range(4)])
        self.play(Transform(nodes[0], new_nodes[0]),
                  Transform(nodes[1], new_nodes[1]),
                  Transform(nodes[2], new_nodes[2]),
                  Transform(nodes[3], new_nodes[3]))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)