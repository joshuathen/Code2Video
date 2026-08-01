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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup layout
        lines = [
            'When Alice and Bob meet, their phones exchange IDs.',
            'These IDs are stored only on their local devices.',
            'No central server is notified of this brief encounter.'
        ]
        self.setup_layout("Step 2: The Digital Handshake (Local Storage)", lines)
        
        # Colors
        BLUE_CLR = "#0000FF"
        BROWN_CLR = "#8B4513"
        GREEN_CLR = "#00FF00"
        
        # Assets
        PHONE_ASSET = "/mmfs1/data/home/jthen/Code2Video/assets/icon/phone.svg"
        SERVER_ASSET = "/mmfs1/data/home/jthen/Code2Video/assets/icon/server.svg"
        
        # === Animation for Lecture Line 1 ===
        # Color change for current line
        self.play(self.lecture[0].animate.set_color(BLUE_CLR))
        
        # Visual Elements: Alice and Bob's phones using SVG assets
        # Alice moved to B3 (Issue 36), Bob at B5 (Issue 38 scaled)
        alice_phone = SVGMobject(PHONE_ASSET, color=WHITE)
        self.place_at_grid(alice_phone, "B3", scale_factor=0.8)
        
        bob_phone = SVGMobject(PHONE_ASSET, color=WHITE)
        self.place_at_grid(bob_phone, "B5", scale_factor=0.8)
        
        alice_label = Text("Alice", font_size=18).next_to(alice_phone, UP, buff=0.1)
        bob_label = Text("Bob", font_size=18).next_to(bob_phone, UP, buff=0.1)
        
        # IDs starting inside phones
        id_a = Text("A1B2", font_size=18, color=BLUE_CLR)
        id_b = Text("B3C4", font_size=18, color=BLUE_CLR)
        id_a.move_to(alice_phone.get_center())
        id_b.move_to(bob_phone.get_center())
        
        # Blue Arrows showing exchange
        arrow_up = CurvedArrow(self.grid["B3"] + RIGHT*0.4, self.grid["B5"] + LEFT*0.4, angle=-TAU/8, color=BLUE_CLR)
        arrow_down = CurvedArrow(self.grid["B5"] + LEFT*0.4, self.grid["B3"] + RIGHT*0.4, angle=-TAU/8, color=BLUE_CLR)
        
        self.play(FadeIn(alice_phone), FadeIn(bob_phone), Write(alice_label), Write(bob_label))
        self.play(Write(id_a), Write(id_b))
        self.play(Create(arrow_up), Create(arrow_down))
        
        # IDs exchanging positions - move slightly below phones for visibility
        self.play(
            id_a.animate.move_to(bob_phone.get_center() + DOWN*0.6),
            id_b.animate.move_to(alice_phone.get_center() + DOWN*0.6),
            run_time=2
        )
        self.wait(0.5)
        
        # === Animation for Lecture Line 2 ===
        # Color change for current line
        self.play(self.lecture[1].animate.set_color(BROWN_CLR))
        
        # Local Log (Digital Shoebox) at E5 (Issue 38 scaled)
        local_log = Rectangle(height=0.8, width=1.4, color=BROWN_CLR, fill_opacity=0.2)
        self.place_at_grid(local_log, "E5", scale_factor=0.8)
        log_label = Text("Seen Today (Log)", font_size=16, color=BROWN_CLR).next_to(local_log, DOWN, buff=0.1)
        
        self.play(Create(local_log), Write(log_label))
        
        # Bob stores Alice's ID locally
        self.play(id_a.animate.move_to(local_log.get_center()).scale(0.8), run_time=1.5)
        self.wait(0.5)
        
        # === Animation for Lecture Line 3 ===
        # Color change for current line
        self.play(self.lecture[2].animate.set_color(GREEN_CLR))
        
        # Central Server icon using SVG asset at E3 (Issue 37)
        server_icon = SVGMobject(SERVER_ASSET, color=GREEN_CLR)
        self.place_at_grid(server_icon, "E3", scale_factor=0.8)
        server_label = Text("Central Server", font_size=16, color=GREEN_CLR).next_to(server_icon, DOWN, buff=0.1)
        
        # Red X over server to show it knows nothing
        cross = VGroup(
            Line(server_icon.get_corner(UL), server_icon.get_corner(DR), color=RED, stroke_width=6),
            Line(server_icon.get_corner(UR), server_icon.get_corner(DL), color=RED, stroke_width=6)
        )
        
        self.play(FadeIn(server_icon), Write(server_label))
        self.play(Create(cross))
        
        self.wait(2)
