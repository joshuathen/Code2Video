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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Trap: Testing n=6", [
            "Test the pattern for six points.", 
            "Expect thirty-two, but count thirty-one.", 
            "The pattern fails at the sixth step."
        ])
        
        # Load SVG
        hexagon_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hexagon.svg", color=WHITE)
        
        # Elements
        nodes = VGroup(*[Dot(color=WHITE) for _ in range(6)])
        circle = Circle(radius=1.5, color=GREY)
        
        # Layout hexagon nodes
        for i, node in enumerate(nodes):
            node.move_to(circle.point_from_proportion(i/6))
            
        hexagon_group = VGroup(circle, nodes, hexagon_icon)
        
        # Applying requested scaling/positioning
        self.place_in_area(hexagon_group, 'C2', 'E4', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        # Test the pattern for six points.
        self.play(FadeIn(hexagon_group))
        self.lecture[0].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Expect thirty-two, but count thirty-one.
        edges = VGroup()
        for i in range(6):
            for j in range(i + 1, 6):
                line = Line(nodes[i].get_center(), nodes[j].get_center(), color="#FF00FF", stroke_width=2)
                edges.add(line)
        
        self.play(Create(edges))
        self.lecture[1].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The pattern fails at the sixth step.
        intersection = Dot(color="#FF0000", radius=0.1)
        intersection.move_to(hexagon_group.get_center())
        
        red_point_label = Text("Intersection", font_size=16, color="#FF0000")
        self.place_at_grid(red_point_label, 'D4', scale_factor=0.5)
        
        self.play(FadeIn(intersection), Write(red_point_label))
        self.lecture[2].set_color("#FF0000")
        self.wait(1)
