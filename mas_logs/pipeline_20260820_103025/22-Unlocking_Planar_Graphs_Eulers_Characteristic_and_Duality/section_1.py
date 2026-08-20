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
        lecture_lines = [
            "A planar graph has no crossing edges.", 
            "Vertices and edges define bounded regions, or faces.", 
            "Don't forget the infinite face surrounding it."
        ]
        self.setup_layout("Prerequisite: Defining the Planar Graph", lecture_lines)
        
        # Asset usage
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg"
        
        # Define graph elements (square shape)
        nodes = VGroup(*[SVGMobject(asset_path).set_color(WHITE) for _ in range(4)])
        edges = VGroup(*[Line(ORIGIN, ORIGIN, color="#FF00FF") for _ in range(4)])
        
        # The "Square Shape" requested by feedback
        graph = VGroup(nodes, edges)
        self.place_in_area(graph, 'B2', 'E5', scale_factor=1.2)
        
        # Manually link edges for the graph
        edges[0].put_start_and_end_on(nodes[0].get_center(), nodes[1].get_center())
        edges[1].put_start_and_end_on(nodes[1].get_center(), nodes[2].get_center())
        edges[2].put_start_and_end_on(nodes[2].get_center(), nodes[3].get_center())
        edges[3].put_start_and_end_on(nodes[3].get_center(), nodes[0].get_center())
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(graph))
        self.lecture[0].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Mic icons as requested by feedback
        mic_icons = VGroup(*[SVGMobject(asset_path).set_color(WHITE) for _ in range(4)])
        self.place_at_grid(mic_icons[0], 'B2', scale_factor=0.5)
        self.place_at_grid(mic_icons[1], 'B5', scale_factor=0.5)
        self.place_at_grid(mic_icons[2], 'E2', scale_factor=0.5)
        self.place_at_grid(mic_icons[3], 'E5', scale_factor=0.5)
        
        self.play(FadeIn(mic_icons))
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)
        
        # Decompose graph (move nodes slightly apart)
        self.play(
            nodes[0].animate.shift(UP * 0.2 + LEFT * 0.2),
            nodes[1].animate.shift(UP * 0.2 + RIGHT * 0.2),
            nodes[2].animate.shift(DOWN * 0.2 + RIGHT * 0.2),
            nodes[3].animate.shift(DOWN * 0.2 + LEFT * 0.2),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Circle shape for infinite face
        circle_shape = Circle(color="#00FFFF")
        self.place_in_area(circle_shape, 'B3', 'E4', scale_factor=0.9)
        circle_shape.set_stroke(width=4)
        
        self.play(Create(circle_shape))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
