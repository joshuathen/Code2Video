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
        self.setup_layout(
            "The Privacy Dilemma: The Health vs. Privacy Trade-off",
            [
                "Tracking virus spread is vital for public health safety.",
                "Standard GPS tracking risks creating a mass surveillance state.",
                "DP-3T offers contact tracing without sacrificing individual privacy."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Show a network of blue circles (#0000FF) with red lines (#FF0000) appearing between them to show spread.
        self.lecture[0].set_color(BLUE)
        
        node_positions = ['C2', 'C5', 'E2', 'E5', 'D3', 'B4', 'F3']
        nodes = VGroup(*[Circle(radius=0.15, color=BLUE, fill_opacity=1) for _ in node_positions])
        for node, pos in zip(nodes, node_positions):
            self.place_at_grid(node, pos)
            
        connections = VGroup(
            Line(nodes[0].get_center(), nodes[4].get_center(), color=RED),
            Line(nodes[4].get_center(), nodes[1].get_center(), color=RED),
            Line(nodes[4].get_center(), nodes[2].get_center(), color=RED),
            Line(nodes[2].get_center(), nodes[6].get_center(), color=RED),
            Line(nodes[3].get_center(), nodes[1].get_center(), color=RED),
            Line(nodes[5].get_center(), nodes[1].get_center(), color=RED)
        )
        
        self.play(Create(nodes), run_time=1)
        self.play(Create(connections), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A large white eye icon (#FFFFFF) [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/eye.svg]
        # appears at the top, sending yellow beams (#FFFF00) to every blue circle.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        eye = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/eye.svg", color=WHITE)
        self.place_in_area(eye, 'A2', 'A5', scale_factor=0.6)
        
        beams = VGroup(*[
            Line(eye.get_bottom(), node.get_center(), color=YELLOW, stroke_width=2)
            for node in nodes
        ])
        
        self.play(FadeIn(eye))
        self.play(Create(beams), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A green shield (#00FF00) [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/shield.svg] 
        # appears around each circle, and the yellow beams from the eye bounce off.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        
        shields = VGroup(*[
            SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/shield.svg", color=GREEN).scale(0.2)
            for _ in nodes
        ])
        for shield, node in zip(shields, nodes):
            shield.move_to(node.get_center())
            
        # Animation for bouncing: move beams back to eye or just shorten them
        bouncing_beams = VGroup(*[
            Line(node.get_center(), eye.get_bottom(), color=YELLOW, stroke_width=2).set_opacity(0.5)
            for node in nodes
        ])
        
        self.play(FadeIn(shields))
        self.play(
            beams.animate.scale(0.1, about_point=eye.get_bottom()).set_opacity(0),
            run_time=1.5
        )
        self.wait(2)
        self.lecture[2].set_color(WHITE)
