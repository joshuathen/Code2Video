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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Prerequisite Knowledge: Bluetooth BLE and Hashing", 
            [
                "Bluetooth Low Energy enables phones to detect nearby devices.",
                "Cryptographic hashes transform data into unique digital fingerprints.",
                "These hashes are one-way, making them impossible to reverse."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Smartphone icon (#FFFFFF) broadcasts blue pulsing 'BLE' circles (#0000FF).
        self.lecture[0].set_color("#0000FF")
        
        # Simple Smartphone Construction
        phone_body = RoundedRectangle(corner_radius=0.1, height=1.6, width=0.9, color=WHITE, fill_opacity=1)
        phone_screen = Rectangle(height=1.2, width=0.7, color=BLACK, fill_opacity=1).move_to(phone_body.get_center()).shift(UP*0.1)
        phone_button = Circle(radius=0.06, color=BLACK, fill_opacity=1).move_to(phone_body.get_center()).shift(DOWN*0.65)
        phone = VGroup(phone_body, phone_screen, phone_button)
        
        # Issue 35 fix: Shifted to C4 and scaled to 1.0
        self.place_at_grid(phone, "C4", scale_factor=1.0)
        
        # Pulsing BLE Circles
        pulses = VGroup(*[
            Circle(radius=0.1, color="#0000FF", stroke_width=4).move_to(phone.get_center()) 
            for _ in range(3)
        ])
        
        self.play(FadeIn(phone))
        self.play(
            LaggedStart(
                *[Succession(
                    FadeIn(p, run_time=0.2),
                    p.animate(run_time=1.2, rate_func=linear).scale(10).set_stroke(opacity=0)
                ) for p in pulses],
                lag_ratio=0.5
            ),
            run_time=2.5
        )
        self.wait(1)
        self.play(FadeOut(phone), FadeOut(pulses))

        # === Animation for Lecture Line 2 ===
        # A yellow 'Secret Key' (#FFFF00) enters a gray 'Hash' machine (#CCCCCC).
        self.lecture[1].set_color("#FFFF00")
        
        # Hash Machine
        machine_box = RoundedRectangle(corner_radius=0.2, height=1.6, width=2.2, color="#CCCCCC", fill_opacity=1)
        machine_label = Text("HASH FUNCTION", color=BLACK, font_size=18).move_to(machine_box.get_center())
        machine = VGroup(machine_box, machine_label)
        # Issue 34 fix: Shifted machine area to C3-E5
        self.place_in_area(machine, "C3", "E5", scale_factor=0.9)
        
        # Secret Key
        secret_key = Text("SECRET KEY", color="#FFFF00", font_size=20)
        # Issue 33 fix: Shifted secret_key to C2, scale 0.8
        self.place_at_grid(secret_key, "C2", scale_factor=0.8)
        
        self.play(FadeIn(machine))
        self.play(Write(secret_key))
        
        # Entry animation: Secret key moves into the machine
        self.play(
            secret_key.animate.move_to(machine.get_center()).scale(0.2).set_opacity(0),
            run_time=1.5
        )
        
        # Process effect (wiggle)
        self.play(machine.animate(rate_func=wiggle).shift(UP*0.05), run_time=0.5)

        # === Animation for Lecture Line 3 ===
        # A unique 'Digital Fingerprint' string (#00FF00) emerges from the machine.
        self.lecture[2].set_color("#00FF00")
        
        fingerprint = Text("a7b8...92f", color="#00FF00", font_size=22)
        fingerprint.move_to(machine.get_center())
        
        # Emergence: Resulting hash moves to the right to C6
        self.play(
            fingerprint.animate.move_to(self.grid["C6"]).scale(1.2),
            run_time=1.5
        )
        
        # One-way Visualization: Impossible to reverse
        # Start from new fingerprint position (C6), end at new key position (C2)
        reverse_arrow = Arrow(start=self.grid["C6"], end=self.grid["C2"], color=RED, buff=0.3)
        cross_line1 = Line(UP+LEFT, DOWN+RIGHT, color=RED, stroke_width=8).scale(0.4)
        cross_line2 = Line(UP+RIGHT, DOWN+LEFT, color=RED, stroke_width=8).scale(0.4)
        # Place cross at the center of the machine
        cross = VGroup(cross_line1, cross_line2).move_to(machine.get_center())
        
        self.play(Create(reverse_arrow))
        self.play(Create(cross))
        
        self.wait(3)
