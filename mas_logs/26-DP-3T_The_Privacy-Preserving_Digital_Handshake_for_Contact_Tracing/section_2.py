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
        # Setup title and lines
        self.setup_layout(
            "Prerequisite Knowledge: Hashing & Bluetooth Beacons", 
            [
                "Hashing transforms data into unique, irreversible digital fingerprints.", 
                "Bluetooth beacons estimate distance without using GPS tracking.", 
                "These tools form the foundation of private contact tracing."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Hash Function Box
        hash_box = Rectangle(width=2.5, height=1.5, color="#696969", fill_opacity=0.3)
        self.place_in_area(hash_box, "B2", "C5")
        hash_label = Text("Hash Function", font_size=20, color="#696969")
        hash_label.next_to(hash_box, UP, buff=0.1)
        
        # Secret Input
        secret_text = Text("Secret", font_size=24, color="#FFFFFF")
        self.place_at_grid(secret_text, "B1")
        
        # Hex Output
        hex_output = Text("7d2...f0", font_size=24, color="#00FF00")
        self.place_at_grid(hex_output, "B6")
        
        self.play(Create(hash_box), Write(hash_label))
        self.play(secret_text.animate.move_to(hash_box.get_center()), run_time=1.5)
        self.play(secret_text.animate.set_opacity(0), hash_box.animate.set_fill(opacity=0.8), run_time=0.5)
        self.play(ReplacementTransform(hash_box.copy().set_opacity(0), hex_output), run_time=1)
        self.play(hash_box.animate.set_fill(opacity=0.3))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Smartphone Icon
        phone_body = RoundedRectangle(corner_radius=0.1, height=1.5, width=0.8, color="#87CEEB", fill_opacity=0.5)
        phone_screen = Rectangle(height=1.1, width=0.6, color=WHITE, fill_opacity=0.1).move_to(phone_body.get_center())
        phone = VGroup(phone_body, phone_screen)
        self.place_at_grid(phone, "E3")
        
        # Bluetooth Label
        rssi_label = Text("Bluetooth RSSI", font_size=20, color="#1E90FF")
        self.place_at_grid(rssi_label, "E5")
        
        self.play(FadeIn(phone))
        self.play(Write(rssi_label))
        
        # Expanding Circles (Signal)
        circles = VGroup(*[
            Circle(radius=r, color="#1E90FF", stroke_opacity=1 - (r/2)) 
            for r in [0.2, 0.5, 0.8, 1.1]
        ])
        circles.move_to(phone.get_center())
        
        def update_circles(group, alpha):
            for i, circle in enumerate(group):
                # Offset each circle's expansion
                effective_alpha = (alpha + i/len(group)) % 1
                circle.set_width((0.1 + effective_alpha * 2.5) * 2)
                circle.set_stroke(opacity=1 - effective_alpha)

        # Animation of signal expansion
        self.play(UpdateFromAlphaFunc(circles, update_circles), run_time=3, rate_func=linear)
        self.add(circles)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Final highlight - subtle pulse of both concepts
        self.play(
            hash_box.animate.scale(1.1),
            phone.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)
