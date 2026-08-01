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
        self.setup_layout("The Privacy Dilemma", [
            'Digital contact tracing helps stop viral spread quickly.',
            'Centralized systems track everyone on a single server.',
            'DP-3T uses decentralization to protect individual privacy.'
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # People nodes
        node_positions = ["B2", "B5", "D3", "D5", "E2"]
        nodes = VGroup()
        for pos in node_positions:
            node = Dot(radius=0.15, color=BLUE_D)
            self.place_at_grid(node, pos)
            nodes.add(node)
            
        # Social connections
        edges = VGroup(
            Line(nodes[0].get_center(), nodes[2].get_center(), stroke_width=2, color=GREY_B),
            Line(nodes[1].get_center(), nodes[2].get_center(), stroke_width=2, color=GREY_B),
            Line(nodes[2].get_center(), nodes[3].get_center(), stroke_width=2, color=GREY_B),
            Line(nodes[2].get_center(), nodes[4].get_center(), stroke_width=2, color=GREY_B)
        )
        
        self.play(Create(nodes), Create(edges))
        
        # Virus spread
        virus = Dot(radius=0.1, color="#FF0000")
        virus.move_to(nodes[0].get_center())
        self.play(FadeIn(virus))
        self.play(virus.animate.move_to(nodes[2].get_center()), run_time=1.5)
        self.play(nodes[2].animate.set_color("#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Server Icon
        server_box = Rectangle(height=1.0, width=1.5, color="#FFFFFF")
        server_label = Text("SERVER", font_size=16, color="#FFFFFF")
        server = VGroup(server_box, server_label)
        self.place_in_area(server, "C3", "D4")
        
        # Connections to server
        server_edges = VGroup()
        for node in nodes:
            server_edges.add(Line(node.get_center(), server.get_center(), color=WHITE, stroke_opacity=0.4))
            
        self.play(
            FadeOut(edges),
            FadeOut(virus),
            FadeIn(server),
            Create(server_edges)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Remove central server and restore local connections
        self.play(
            FadeOut(server),
            FadeOut(server_edges),
            FadeIn(edges),
            nodes[2].animate.set_color(BLUE_D) # Reset color to show privacy/anonymity
        )
        
        # Show signal waves (Broadcast effect)
        self.play(
            Broadcast(nodes[0], focal_point=nodes[0].get_center(), color=YELLOW),
            Broadcast(nodes[2], focal_point=nodes[2].get_center(), color=YELLOW),
            Broadcast(nodes[4], focal_point=nodes[4].get_center(), color=YELLOW),
            run_time=2
        )
        self.wait(2)
