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
        # Titles and lecture lines (aligned with Stage-3 prompt requirements)
        title = "The Conflict: Health vs. Privacy"
        lines = [
            'Meet Alice and Bob, two privacy-conscious friends.', 
            "Traditional tracing often requires a central authority's surveillance.", 
            'DP-3T uses a privacy shield to block central tracking.'
        ]
        
        self.setup_layout(title, lines)

        # Colors
        ALICE_COLOR = "#58D68D"
        BOB_COLOR = "#F5B041"
        SERVER_COLOR = "#EC7063"
        SHIELD_COLOR = "#5DADE2"

        # === Animation for Lecture Line 1 ===
        # Meet Alice and Bob, two privacy-conscious friends.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        alice_node = Circle(radius=0.5, color=ALICE_COLOR, fill_opacity=0.3)
        self.place_at_grid(alice_node, "D3") # Fix Issue 32
        
        bob_node = Circle(radius=0.5, color=BOB_COLOR, fill_opacity=0.3)
        self.place_at_grid(bob_node, "D6") # Fix Issue 33
        
        alice_label = Text("Alice", font_size=18, color=ALICE_COLOR)
        bob_label = Text("Bob", font_size=18, color=BOB_COLOR)
        
        # Using next_to relative to placed nodes is allowed for labels
        alice_label.next_to(alice_node, DOWN, buff=0.2)
        bob_label.next_to(bob_node, DOWN, buff=0.2)
        
        self.play(Create(alice_node), Create(bob_node))
        self.play(Write(alice_label), Write(bob_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Traditional tracing often requires a central authority's surveillance.
        self.play(self.lecture[1].animate.set_color(SERVER_COLOR))
        
        # Central Server icon
        server_rect = RoundedRectangle(corner_radius=0.1, height=1, width=1.5, color=SERVER_COLOR, fill_opacity=0.5)
        server_txt = Text("SERVER", font_size=16, color=WHITE)
        server_group = VGroup(server_rect, server_txt)
        self.place_in_area(server_group, "A4", "A5") # Fix Issue 34
        
        # Surveillance lines
        line_a = Line(alice_node.get_top(), server_group.get_bottom(), color=SERVER_COLOR, stroke_width=2)
        line_b = Line(bob_node.get_top(), server_group.get_bottom(), color=SERVER_COLOR, stroke_width=2)
        
        data_label_a = Text("Location Data", font_size=12, color=SERVER_COLOR).next_to(line_a, LEFT, buff=0.1)
        data_label_b = Text("Location Data", font_size=12, color=SERVER_COLOR).next_to(line_b, RIGHT, buff=0.1)

        self.play(FadeIn(server_group))
        self.play(Create(line_a), Create(line_b))
        self.play(Write(data_label_a), Write(data_label_b))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # DP-3T uses a privacy shield to block central tracking.
        self.play(self.lecture[2].animate.set_color(SHIELD_COLOR))
        
        # Privacy Shield
        shield = Polygon(
            [-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.2, 0], [0, -0.6, 0], [-0.5, -0.2, 0],
            color=SHIELD_COLOR, fill_opacity=0.8
        )
        self.place_in_area(shield, "B4", "B5", scale_factor=0.8) # Fix Issue 34
        
        # Blocking lines
        self.play(FadeIn(shield))
        self.play(
            FadeOut(line_a), FadeOut(line_b),
            FadeOut(data_label_a), FadeOut(data_label_b)
        )
        self.wait(2)
