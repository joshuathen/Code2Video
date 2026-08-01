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
        # Setup layout
        lines = [
            'DP-3T ensures matching happens only on your device.',
            'The server never learns who met whom.',
            'Cryptography secures health data against surveillance.'
        ]
        self.setup_layout("Summary: Privacy by Design", lines)

        # Helper to create a lock
        def get_lock(color="#FFD700"):
            body = Square(side_length=0.2, fill_opacity=1, color=color, stroke_width=1)
            shackle = Arc(radius=0.08, start_angle=0, angle=PI, color=color, stroke_width=2).shift(UP*0.1)
            return VGroup(body, shackle)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        
        # Centralized Web
        cent_color = "#FF4444"
        cent_server = Dot(color=cent_color, radius=0.15)
        self.place_at_grid(cent_server, "C2")
        
        cent_nodes = VGroup(*[Dot(color=cent_color, radius=0.1) for _ in range(4)])
        self.place_at_grid(cent_nodes[0], "B1", scale_factor=0.7)
        self.place_at_grid(cent_nodes[1], "B3")
        self.place_at_grid(cent_nodes[2], "D1", scale_factor=0.7)
        self.place_at_grid(cent_nodes[3], "D3")
        
        cent_lines = VGroup(*[Line(cent_server.get_center(), node.get_center(), color=cent_color, stroke_width=2) for node in cent_nodes])
        
        cent_label = Text("Centralized", font_size=18, color=cent_color)
        self.place_in_area(cent_label, "A1", "A3")
        
        # Decentralized Dots
        dec_color = "#00FF00"
        dec_nodes = VGroup(*[Dot(color=dec_color, radius=0.1) for _ in range(4)])
        self.place_at_grid(dec_nodes[0], "B5")
        self.place_at_grid(dec_nodes[1], "C5", scale_factor=0.8)
        self.place_at_grid(dec_nodes[2], "D6")
        self.place_at_grid(dec_nodes[3], "E5")
        
        dec_label = Text("DP-3T (Local)", font_size=18, color=dec_color)
        self.place_in_area(dec_label, "A4", "A6")
        
        # Divider line
        divider = DashedLine(self.grid["A4"] + LEFT*0.5 + UP*0.5, self.grid["F4"] + LEFT*0.5 + DOWN*0.5, color=GREY)

        self.play(
            FadeIn(cent_label), FadeIn(cent_server), Create(cent_lines), FadeIn(cent_nodes),
            FadeIn(dec_label), FadeIn(dec_nodes),
            Create(divider)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        
        # Handshakes in decentralized area
        handshake1 = Line(dec_nodes[0].get_center(), dec_nodes[1].get_center(), color=dec_color, stroke_width=2)
        handshake2 = Line(dec_nodes[2].get_center(), dec_nodes[3].get_center(), color=dec_color, stroke_width=2)
        
        lock1 = get_lock("#FFD700")
        lock2 = get_lock("#FFD700")
        lock1.move_to(handshake1.get_center())
        lock2.move_to(handshake2.get_center())

        self.play(
            Create(handshake1), Create(handshake2),
            FadeIn(lock1), FadeIn(lock2),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        privacy_text = Text("Privacy by Design", color="#00FF00", font_size=32)
        self.place_in_area(privacy_text, "F2", "F5", scale_factor=0.8)
        
        # Glow and scale effect
        self.play(
            Write(privacy_text),
            privacy_text.animate.scale(1.3),
            Indicate(privacy_text, color="#00FF00", scale_factor=1.1),
            run_time=2
        )
        self.wait(2)
