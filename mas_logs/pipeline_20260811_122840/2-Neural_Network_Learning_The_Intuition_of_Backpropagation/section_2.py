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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Computational Graph", [
            "Neural networks are computational graphs of flow.",
            "Nodes process data; edges represent weights.",
            "Adjust weights to change the final output."
        ])
        
        # Load assets
        node_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/node.svg")
        edge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/edge.svg")

        # Create nodes using assets
        nodes = VGroup(*[node_icon.copy() for _ in range(6)])
        
        # Grid positioning per feedback (Issues 29, 30, 31)
        self.place_at_grid(nodes[0], 'C2', scale_factor=0.6)
        self.place_at_grid(nodes[1], 'E2', scale_factor=0.6)
        self.place_at_grid(nodes[2], 'C3', scale_factor=0.6)
        self.place_at_grid(nodes[3], 'E3', scale_factor=0.6)
        self.place_at_grid(nodes[4], 'C4', scale_factor=0.6)
        self.place_at_grid(nodes[5], 'E4', scale_factor=0.6)

        # Labels
        label_net = Text("Network", font_size=20)
        label_net.next_to(nodes[2], UP, buff=0.2).scale(0.75)
        
        # Edges
        edges = VGroup()
        for i in [0, 2, 4]:
            for j in [1, 3, 5]:
                # Simple line representation using asset reference implies logical link
                e = Line(nodes[i].get_right(), nodes[j].get_left(), color=GRAY, stroke_width=2)
                edges.add(e)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(nodes), FadeIn(edges), Write(label_net))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#8A2BE2"))
        
        # Animate forward flow with Purple color as per storyboard
        flow_lines = VGroup(*[Line(e.get_start(), e.get_end(), color="#8A2BE2", stroke_width=4) for e in edges])
        self.play(Create(flow_lines), run_time=2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        # Animate Fade of one edge/node connection
        self.play(FadeOut(edges[0]), FadeOut(flow_lines[0]), run_time=1.5)
        self.wait(1)
