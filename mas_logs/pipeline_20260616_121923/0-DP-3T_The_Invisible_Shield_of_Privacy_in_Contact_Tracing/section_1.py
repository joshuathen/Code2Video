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
        title_text = "The Privacy Paradox: Contact Tracing vs. Surveillance"
        lines = [
            "During pandemics, we need to track viral spread quickly.",
            "Central tracking risks individual location privacy and surveillance.",
            "DP-3T offers a privacy-first approach to digital contact tracing.",
            "Can we notify exposure without a central authority knowing?",
            "Meet Alice and Bob in our contact tracing example."
        ]
        
        self.setup_layout(title_text, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        park = Rectangle(width=5.5, height=2.5, color=GREEN_E, fill_opacity=0.2, stroke_width=1)
        self.place_in_area(park, "B1", "C6")
        
        # Asset: Alice and Bob icons [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/user.svg] (#FFFFFF)
        alice_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/user.svg", color=WHITE).set_stroke(width=1)
        alice_label = Text("Alice", font_size=18, color=WHITE)
        alice = VGroup(alice_icon, alice_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(alice, "B2", scale_factor=0.5)

        bob_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/user.svg", color=WHITE).set_stroke(width=1)
        bob_label = Text("Bob", font_size=18, color=WHITE)
        bob = VGroup(bob_icon, bob_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(bob, "B5", scale_factor=0.5)

        self.play(FadeIn(park), FadeIn(alice), FadeIn(bob))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(RED)
        
        virus = Star(n=7, color="#FF0000", fill_opacity=1, stroke_width=0).scale(0.2)
        self.place_in_area(virus, "B3", "B4")
        
        self.play(FadeIn(virus))
        self.play(Flash(virus, color="#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(GREEN)
        
        server = VGroup(
            Square(side_length=0.8, color="#00FF00", fill_opacity=0.1),
            Text("Central Server", font_size=14, color="#00FF00")
        ).arrange(DOWN, buff=0.1)
        # Fix: Move server to E3-E4 for symmetry
        self.place_in_area(server, 'E3', 'E4', scale_factor=0.9)
        
        # Tracking lines
        trace_a = Line(alice.get_center(), server.get_center(), color=RED, stroke_width=1).set_opacity(0.6)
        trace_b = Line(bob.get_center(), server.get_center(), color=RED, stroke_width=1).set_opacity(0.6)
        
        self.play(FadeIn(server), Create(trace_a), Create(trace_b))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(GRAY)
        self.lecture[3].set_color(BLUE)
        
        # Privacy Wall
        wall = Line(self.grid["D1"], self.grid["D6"], color="#0000FF", stroke_width=10)
        wall_text = Text("Privacy Wall", font_size=20, color="#0000FF")
        # Fix: Place wall text in area D3-D4 to avoid overlap with line
        self.place_in_area(wall_text, 'D3', 'D4', scale_factor=0.8)
        wall_text.shift(UP * 0.4) # Keep slight shift to clear the thick line
        
        self.play(Create(wall), Write(wall_text))
        
        # Block tracking lines
        self.play(
            trace_a.animate.set_color(BLUE).set_points_as_corners([alice.get_center(), self.grid["D2"]]),
            trace_b.animate.set_color(BLUE).set_points_as_corners([bob.get_center(), self.grid["D5"]])
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(GRAY)
        self.lecture[4].set_color(WHITE)
        
        banner_text = Text("Privacy-Preserving Contact Tracing", font_size=24, color=WHITE)
        # Fix: Scale banner text to 0.7 to avoid visual crowding
        self.place_in_area(banner_text, 'A1', 'A6', scale_factor=0.7)
        
        self.play(Write(banner_text))
        self.play(Indicate(banner_text, color=WHITE))
        self.wait(2)
