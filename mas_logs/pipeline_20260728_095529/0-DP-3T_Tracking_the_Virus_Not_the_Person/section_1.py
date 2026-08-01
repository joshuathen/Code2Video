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
            "The Privacy Paradox: The Contact Tracing Challenge",
            [
                "Public health needs to track virus spread efficiently.",
                "But surveillance states threaten our fundamental right to privacy.",
                "DP-3T solves this by decentralizing all contact data.",
                "Bob learns of exposure without anyone knowing his location."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        health_icon = VGroup(
            Line(LEFT, RIGHT, color="#00FF00", stroke_width=8),
            Line(UP, DOWN, color="#00FF00", stroke_width=8)
        )
        # Resolved Issue 22: placement A1-B2, scale 0.5
        self.place_in_area(health_icon, "A1", "B2", scale_factor=0.5)
        
        privacy_icon = VGroup(
            RoundedRectangle(corner_radius=0.1, width=1, height=0.7, color="#FF00FF"),
            Arc(radius=0.3, start_angle=0, angle=PI, color="#FF00FF").shift(UP*0.35)
        )
        # Resolved Issue 23: placement E4-F6, scale 0.6
        self.place_in_area(privacy_icon, "E4", "F6", scale_factor=0.6)
        
        divider = Line(self.grid["A3"] + RIGHT*0.4 + UP*0.4, self.grid["F3"] + RIGHT*0.4 + DOWN*0.4, color=WHITE)
        
        self.play(Create(health_icon), Create(privacy_icon), Create(divider))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FF0000")
        )
        self.play(FadeOut(health_icon), FadeOut(privacy_icon), FadeOut(divider))
        
        # Red Eye (Asset integration)
        # Resolved Issue 20: Use the SVG asset
        eye = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eye.svg", color="#FF0000")
        self.place_in_area(eye, "A3", "A4", scale_factor=0.8)
        
        alice = Dot(color="#5555FF", radius=0.2)
        alice_label = Text("Alice", font_size=16, color="#5555FF")
        self.place_at_grid(alice, "D2")
        alice_label.next_to(alice, DOWN, buff=0.1)
        
        bob = Dot(color="#FFA500", radius=0.2)
        bob_label = Text("Bob", font_size=16, color="#FFA500")
        self.place_at_grid(bob, "D5")
        bob_label.next_to(bob, DOWN, buff=0.1)
        
        ray1 = Line(eye.get_center(), alice.get_center(), color="#FF0000", stroke_width=2, stroke_opacity=0.5)
        ray2 = Line(eye.get_center(), bob.get_center(), color="#FF0000", stroke_width=2, stroke_opacity=0.5)
        
        self.play(Create(eye), Create(alice), Create(alice_label), Create(bob), Create(bob_label))
        self.play(Create(ray1), Create(ray2))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFFFF")
        )
        
        shield = Rectangle(width=4, height=0.2, color="#FFFFFF", fill_opacity=0.8)
        # Resolved Issue 24: placement C1-C6, scale 0.8
        self.place_in_area(shield, "C1", "C6", scale_factor=0.8)
        
        self.play(Create(shield))
        self.play(
            alice.animate.set_color("#808080"),
            bob.animate.set_color("#808080"),
            FadeOut(ray1),
            FadeOut(ray2)
        )
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#FFFF00")
        )
        
        # Dots move near
        self.play(
            alice.animate.move_to(self.grid["D3"]),
            alice_label.animate.next_to(self.grid["D3"], DOWN, buff=0.1),
            bob.animate.move_to(self.grid["D4"]),
            bob_label.animate.next_to(self.grid["D4"], DOWN, buff=0.1)
        )
        
        distance_arrow = DoubleArrow(
            self.grid["D3"], self.grid["D4"], 
            color="#FFFF00", buff=0.1, tip_length=0.1
        )
        dist_text = Text("Encounter", font_size=14, color="#FFFF00").next_to(distance_arrow, UP, buff=0.1)
        
        self.play(Create(distance_arrow), Create(dist_text))
        self.wait(1)
        
        # Dots move away
        self.play(
            alice.animate.move_to(self.grid["D1"]),
            alice_label.animate.next_to(self.grid["D1"], DOWN, buff=0.1),
            bob.animate.move_to(self.grid["F6"]),
            bob_label.animate.next_to(self.grid["F6"], DOWN, buff=0.1),
            FadeOut(distance_arrow),
            FadeOut(dist_text)
        )
        self.wait(2)
