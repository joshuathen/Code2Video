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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title = "Conclusion: The Decentralized Win (0:30)"
        lines = [
            "DP-3T keeps sensitive data off the central server.",
            "Users remain anonymous dots on a hidden map.",
            "Privacy and public health can coexist through decentralization."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Split screen: Server side and User side
        server_icon = Square(side_length=1.2, color="#FFFFFF", fill_opacity=0.2)
        server_label = Text("Server", font_size=20, color="#FFFFFF")
        server_data = Text("List of Seeds\n[a8f2...]\n[b1c9...]", font_size=16, color="#FFFFFF")
        
        server_group = VGroup(server_icon, server_label, server_data).arrange(DOWN, buff=0.2)
        # Resolved Issue #45: Adjusted scale_factor to 0.8
        self.place_at_grid(server_group, "B2", scale_factor=0.8)

        user_icon = Circle(radius=0.6, color="#3498DB", fill_opacity=0.4)
        user_label = Text("User Device", font_size=20, color="#3498DB")
        user_data = Text("Personal Identity\nAlice / Bob", font_size=16, color="#3498DB")
        
        user_group = VGroup(user_icon, user_label, user_data).arrange(DOWN, buff=0.2)
        # Resolved Issue #44: Adjusted scale_factor to 0.8
        self.place_at_grid(user_group, "B5", scale_factor=0.8)

        self.play(
            self.lecture[0].animate.set_color("#FFFFFF"),
            FadeIn(server_group),
            FadeIn(user_group)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # A grey wall between them
        wall = Rectangle(width=0.1, height=4.5, color="#95A5A6", fill_opacity=0.8)
        self.place_in_area(wall, "A3", "F4", scale_factor=1.0)
        
        # Dots on a map for line 2
        map_bg = Rectangle(width=2.5, height=2.5, color="#34495E", fill_opacity=0.1).set_stroke(opacity=0.3)
        self.place_at_grid(map_bg, "D5", scale_factor=1.0)
        
        dots = VGroup(*[Dot(color="#3498DB", radius=0.08) for _ in range(8)])
        for i, dot in enumerate(dots):
            # Scatter dots randomly around the map area
            offset = np.array([np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8), 0])
            dot.move_to(self.grid["D5"] + offset)

        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#3498DB"),
            Create(wall),
            FadeIn(map_bg)
        )
        self.play(LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=0.1))
        
        # Identity data hitting the wall and bouncing/vanishing
        id_packet = Text("Identity Info", font_size=14, color="#3498DB")
        id_packet.move_to(self.grid["C5"])
        
        self.play(id_packet.animate.move_to(self.grid["C4"]), run_time=1)
        self.play(Indicate(wall, color=RED), FadeOut(id_packet))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        trust_text = Text("Decentralized Trust", font_size=32, color="#2ECC71", weight=BOLD)
        # Resolved Issue #43: Adjusted area to F2-F5 and scale_factor to 1.0 to avoid overlap
        self.place_in_area(trust_text, "F2", "F5", scale_factor=1.0)

        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#2ECC71"),
            Write(trust_text)
        )
        
        # Final glow effect for the whole system
        surround_rect = SurroundingRectangle(VGroup(server_group, user_group, wall, map_bg), color="#2ECC71", buff=0.3)
        self.play(Create(surround_rect))
        self.wait(3)
