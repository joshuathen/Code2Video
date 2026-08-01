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

class Section7Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines
        lecture_lines = [
            "The server never knows who Bob is or met.",
            "Your identity and location stay on your device.",
            "DP-3T balances public safety with individual digital privacy."
        ]
        self.setup_layout("Summary: The Decentralized Advantage", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Colors: Green for shield/security context
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Shield representation
        shield_body = RoundedRectangle(corner_radius=0.1, height=2.2, width=1.8, color="#00FF00", fill_opacity=0.3)
        shield_bottom = Triangle(color="#00FF00", fill_opacity=0.3).rotate(PI).scale(0.9).next_to(shield_body, DOWN, buff=-0.2)
        shield = VGroup(shield_body, shield_bottom)
        self.place_in_area(shield, "C3", "D4", scale_factor=1.0)
        
        # Alice and Bob's phones behind the shield
        alice_phone = VGroup(
            RoundedRectangle(corner_radius=0.05, height=0.8, width=0.45, color=BLUE),
            Line(start=[-0.15, -0.3, 0], end=[0.15, -0.3, 0], stroke_width=2, color=BLUE)
        )
        bob_phone = VGroup(
            RoundedRectangle(corner_radius=0.05, height=0.8, width=0.45, color=RED),
            Line(start=[-0.15, -0.3, 0], end=[0.15, -0.3, 0], stroke_width=2, color=RED)
        )
        
        self.place_at_grid(alice_phone, "C3")
        self.place_at_grid(bob_phone, "D4")
        
        # Display phones first then cover with shield
        self.play(Create(alice_phone), Create(bob_phone))
        self.play(FadeIn(shield))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Colors: White and Orange-Red for key privacy text
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        
        privacy_text = Text("Privacy First", font_size=24, color="#FFFFFF")
        tracking_text = Text("No Central Tracking", font_size=24, color="#FF4500")
        
        self.place_at_grid(privacy_text, "B3")
        self.place_at_grid(tracking_text, "E4")
        
        self.play(Write(privacy_text))
        self.play(Write(tracking_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Colors: Dark gray for server and random data context
        self.play(self.lecture[2].animate.set_color("#A9A9A9"))
        
        server_box = Square(side_length=1.2, color="#A9A9A9", fill_opacity=0.2)
        server_label = Text("Server", font_size=18, color="#A9A9A9").next_to(server_box, UP, buff=0.1)
        server_icon = VGroup(server_box, server_label)
        self.place_at_grid(server_icon, "A6", scale_factor=0.8)
        
        keys_label = Text("Random Keys Only", font_size=16, color="#A9A9A9")
        self.place_at_grid(keys_label, "B6")
        
        key_data = VGroup(
            Text("0x8f2...", font_size=14, color="#A9A9A9"),
            Text("0x1a4...", font_size=14, color="#A9A9A9"),
            Text("0xcc9...", font_size=14, color="#A9A9A9")
        ).arrange(DOWN, buff=0.1).next_to(server_box, DOWN, buff=0.1)
        
        # Show server is isolated from the "handshake" happening behind the shield
        self.play(FadeIn(server_icon), Write(keys_label))
        self.play(FadeIn(key_data))
        
        # Cross mark to show no connection between server and identities
        cross = VGroup(
            Line(start=[-0.5, -0.5, 0], end=[0.5, 0.5, 0], color=RED),
            Line(start=[-0.5, 0.5, 0], end=[0.5, -0.5, 0], color=RED)
        ).scale(0.5)
        self.place_in_area(cross, "B5", "B5") # Place between shield and server
        
        self.play(Create(cross))
        self.wait(2)
