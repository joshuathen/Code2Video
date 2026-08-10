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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Planar graphs have no crossing edges.", "Faces are the enclosed regions.", "Includes the outer infinite face."]
        self.setup_layout("Prerequisite: Defining the Planar Graph", lecture_lines)
        
        # Create a simple graph
        vertices = [0, 1, 2, 3]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        positions = {
            0: self.grid["B2"],
            1: self.grid["B5"],
            2: self.grid["E5"],
            3: self.grid["E2"]
        }
        
        graph = Graph(vertices, edges, layout=positions, vertex_config={"radius": 0.2, "fill_color": WHITE}, edge_config={"stroke_width": 4, "color": WHITE})
        # Applying the fix recommended in issue 22 to improve layout balance and readability
        self.place_in_area(graph, "C3", "E5", scale_factor=0.55)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(graph))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        self.play(graph.animate.set_vertex_fill("#FF5733"), graph.animate.set_edge_color("#33FF57"))
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#33FF57")

        # === Animation for Lecture Line 3 ===
        self.play(graph.animate.set_edge_color("#3357FF"))
        self.play(Flash(graph, color="#3357FF", run_time=1.5))
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#3357FF")
