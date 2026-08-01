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
        # Title and Lecture Lines
        title_text = "The Privacy Dilemma"
        lines = [
            "Public health requires tracking viral spread efficiently.",
            "Centralized systems risk exposing private locations and identities.",
            "DP-3T offers a decentralized, privacy-first alternative."
        ]
        self.setup_layout(title_text, lines)

        # Colors
        ALICE_COLOR = "#FFD700"
        BOB_COLOR = "#87CEEB"
        SERVER_COLOR = "#A9A9A9"
        ALERT_COLOR = "#FF0000"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(ALICE_COLOR)
        
        alice = Circle(radius=0.5, color=ALICE_COLOR, fill_opacity=0.3)
        alice_label = Text("Alice", font_size=18, color=ALICE_COLOR)
        alice_group = VGroup(alice, alice_label.next_to(alice, DOWN, buff=0.1))
        self.place_in_area(alice_group, "C1", "D2")
        
        bob = Circle(radius=0.5, color=BOB_COLOR, fill_opacity=0.3)
        bob_label = Text("Bob", font_size=18, color=BOB_COLOR)
        bob_group = VGroup(bob, bob_label.next_to(bob, DOWN, buff=0.1))
        self.place_in_area(bob_group, "C5", "D6")

        self.play(Create(alice_group), Create(bob_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(SERVER_COLOR)

        server = Square(side_length=0.8, color=SERVER_COLOR, fill_opacity=0.5)
        server_icon_label = Text("Central Server", font_size=16, color=SERVER_COLOR)
        server_group = VGroup(server, server_icon_label.next_to(server, UP, buff=0.1))
        self.place_in_area(server_group, "A3", "B4")

        line_alice = Line(alice.get_top(), server.get_bottom(), color=SERVER_COLOR)
        line_bob = Line(bob.get_top(), server.get_bottom(), color=SERVER_COLOR)
        
        log_label = Text("Location Log", font_size=16, color=WHITE)
        self.place_at_grid(log_label, "B4", scale_factor=0.8) # Positioned near server

        self.play(Create(server_group))
        self.play(Create(line_alice), Create(line_bob), Write(log_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(ALERT_COLOR)

        # Create a large red 'X'
        x_line1 = Line(self.grid["A3"], self.grid["B4"], color=ALERT_COLOR, stroke_width=8)
        x_line2 = Line(np.array([self.grid["A4"][0], self.grid["A3"][1], 0]), 
                       np.array([self.grid["A3"][0], self.grid["B4"][1], 0]), 
                       color=ALERT_COLOR, stroke_width=8)
        red_x = VGroup(x_line1, x_line2)

        self.play(Create(red_x))
        self.wait(0.5)
        
        # Fade away decentralized shift components
        self.play(
            FadeOut(server_group),
            FadeOut(line_alice),
            FadeOut(line_bob),
            FadeOut(log_label),
            FadeOut(red_x)
        )
        
        self.lecture[2].set_color(WHITE)
        self.wait(2)
