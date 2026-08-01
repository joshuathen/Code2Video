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
        self.setup_layout("Summary: Why it Works", [
            "The server only acts as a public key repository.",
            "Matching happens locally, ensuring true privacy by design.",
            "DP-3T proves health tracking doesn't require mass surveillance."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # Server Icon
        server_body = RoundedRectangle(height=1.2, width=0.8, corner_radius=0.1, fill_opacity=1, fill_color="#888888", stroke_color=WHITE)
        server_slots = VGroup(*[Line(LEFT*0.2, RIGHT*0.2, color=BLACK, stroke_width=2).shift(UP*0.3 - i*0.25*UP) for i in range(3)])
        server = VGroup(server_body, server_slots)
        self.place_in_area(server, "B3", "C4")
        
        server_label = Text("Passive Storage", font_size=18, color=WHITE)
        self.place_at_grid(server_label, "D3")

        self.play(Create(server), Write(server_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition lecture line highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(BLUE)
        )

        # Phone Asset
        phone = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg")
        phone.set_color(WHITE)
        self.place_in_area(phone, "E2", "F3", scale_factor=1.2)

        # Magnifying Glass
        mag_circle = Circle(radius=0.4, color="#FFFFFF", stroke_width=4)
        mag_handle = Line(ORIGIN, 0.4 * (RIGHT + DOWN), color="#FFFFFF", stroke_width=6)
        mag_glass = VGroup(mag_circle, mag_handle)
        self.place_in_area(mag_glass, "E2", "F3", scale_factor=0.8)
        mag_glass.shift(UP*0.1 + LEFT*0.1)

        # Checkmarks inside - Use Text with Unicode to avoid LaTeX requirement
        check1 = Text("✓", color=GREEN, font_size=30)
        check2 = Text("✓", color=GREEN, font_size=30)
        self.place_at_grid(check1, "E2", scale_factor=0.6)
        self.place_at_grid(check2, "F3", scale_factor=0.6)

        self.play(FadeIn(phone))
        self.play(Create(mag_glass))
        self.play(Write(check1), Write(check2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition lecture line highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GREEN)
        )

        # Large green checkmark and 'Privacy Secured' text
        big_check = Text("✓", color="#00FF00", font_size=160)
        self.place_in_area(big_check, "B2", "E5")

        privacy_text = Text("Privacy Secured", color="#00FF00", font_size=32)
        self.place_in_area(privacy_text, "E3", "F4")

        # Clear previous icons
        self.play(
            FadeOut(server), 
            FadeOut(server_label), 
            FadeOut(phone), 
            FadeOut(mag_glass), 
            FadeOut(check1), 
            FadeOut(check2)
        )
        self.play(GrowFromCenter(big_check))
        self.play(Write(privacy_text))
        self.wait(2)
