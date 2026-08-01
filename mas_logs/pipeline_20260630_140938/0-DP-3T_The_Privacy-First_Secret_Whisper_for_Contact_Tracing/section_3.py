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

class Section3Scene(TeachingScene):
    def construct(self):
        # Fetching storyboard data
        title = "Step 1: Local Key Generation (The Secret Seed)"
        lines = [
            "Your phone generates a secret key stored only locally.",
            "Every day, it derives a unique Temporary Exposure Key.",
            "From this, it creates Rotating Proximity Identifiers (RPIs).",
            "RPIs change every fifteen minutes to prevent tracking users.",
            "No personal identity is ever linked to these random IDs.",
        ]
        self.setup_layout(title, lines)
        
        # Colors
        BLUE = "#0000FF"
        GOLD = "#FFD700"
        SILVER = "#C0C0C0"
        WHITE = "#FFFFFF"
        YELLOW = "#FFFF00"
        RED = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # Your phone generates a secret key stored only locally.
        self.lecture[0].set_color(BLUE)
        phone = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg", color=BLUE)
        self.place_at_grid(phone, "C1", scale_factor=0.8)
        
        # Gold key inside phone
        gold_key_head = Circle(radius=0.1, color=GOLD, fill_opacity=1)
        gold_key_body = Rectangle(width=0.2, height=0.06, color=GOLD, fill_opacity=1).next_to(gold_key_head, RIGHT, buff=-0.02)
        gold_key = VGroup(gold_key_head, gold_key_body)
        self.place_at_grid(gold_key, "C1", scale_factor=0.6)
        
        self.play(DrawBorderThenFill(phone))
        self.play(FadeIn(gold_key))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Every day, it derives a unique Temporary Exposure Key.
        self.lecture[1].set_color(SILVER)
        
        # Slightly different silver key
        silver_key_head = Circle(radius=0.12, color=SILVER, fill_opacity=1)
        silver_key_body = Rectangle(width=0.25, height=0.07, color=SILVER, fill_opacity=1).next_to(silver_key_head, RIGHT, buff=-0.02)
        silver_key = VGroup(silver_key_head, silver_key_body)
        self.place_at_grid(silver_key, "C3", scale_factor=0.6)
        
        daily_label = Text("Daily Key", font_size=18, color=SILVER)
        self.place_at_grid(daily_label, "B3", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(gold_key.copy(), silver_key),
            FadeIn(daily_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # From this, it creates Rotating Proximity Identifiers (RPIs).
        self.lecture[2].set_color(WHITE)
        
        rpi1 = Dot(color=WHITE)
        rpi2 = Dot(color=WHITE)
        rpi3 = Dot(color=WHITE)
        self.place_at_grid(rpi1, "B5")
        self.place_at_grid(rpi2, "C5")
        self.place_at_grid(rpi3, "D5")
        
        rpi1_label = Text("RPI-1", font_size=14, color=WHITE).next_to(rpi1, RIGHT, buff=0.1)
        rpi2_label = Text("RPI-2", font_size=14, color=WHITE).next_to(rpi2, RIGHT, buff=0.1)
        rpi3_label = Text("RPI-3", font_size=14, color=WHITE).next_to(rpi3, RIGHT, buff=0.1)

        line1 = Line(silver_key.get_right(), rpi1.get_center(), color=WHITE, stroke_width=1)
        line2 = Line(silver_key.get_right(), rpi2.get_center(), color=WHITE, stroke_width=1)
        line3 = Line(silver_key.get_right(), rpi3.get_center(), color=WHITE, stroke_width=1)

        self.play(
            LaggedStart(
                AnimationGroup(Create(line1), FadeIn(rpi1), FadeIn(rpi1_label)),
                AnimationGroup(Create(line2), FadeIn(rpi2), FadeIn(rpi2_label)),
                AnimationGroup(Create(line3), FadeIn(rpi3), FadeIn(rpi3_label)),
                lag_ratio=0.3
            )
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # RPIs change every fifteen minutes to prevent tracking users.
        self.lecture[3].set_color(YELLOW)
        
        self.play(
            rpi1.animate.set_opacity(0.3),
            rpi1_label.animate.set_opacity(0.3),
            rpi2.animate.set_color(YELLOW).scale(1.5),
            rpi2_label.animate.set_color(YELLOW).scale(1.2),
            run_time=1
        )
        self.wait(0.5)
        self.play(
            rpi2.animate.set_opacity(0.3).set_color(WHITE).scale(1/1.5),
            rpi2_label.animate.set_opacity(0.3).set_color(WHITE).scale(1/1.2),
            rpi3.animate.set_color(YELLOW).scale(1.5),
            rpi3_label.animate.set_color(YELLOW).scale(1.2),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # No personal identity is ever linked to these random IDs.
        self.lecture[4].set_color(WHITE)
        
        # Reset RPI highlighting for clarity
        self.play(
            rpi3.animate.set_color(WHITE).scale(1/1.5).set_opacity(1.0),
            rpi3_label.animate.set_color(WHITE).scale(1/1.2).set_opacity(1.0),
            rpi1.animate.set_opacity(1.0),
            rpi1_label.animate.set_opacity(1.0),
            rpi2.animate.set_opacity(1.0),
            rpi2_label.animate.set_opacity(1.0),
        )

        person_head = Circle(radius=0.15, color=WHITE, fill_opacity=1)
        person_body = Polygon([-0.2, -0.35, 0], [0.2, -0.35, 0], [0.15, 0, 0], [-0.15, 0, 0], color=WHITE, fill_opacity=1)
        person = VGroup(person_head, person_body).arrange(DOWN, buff=0.05)
        self.place_at_grid(person, "C6", scale_factor=0.8)
        
        cross = VGroup(
            Line([-0.2, -0.2, 0], [0.2, 0.2, 0], color=RED, stroke_width=6),
            Line([-0.2, 0.2, 0], [0.2, -0.2, 0], color=RED, stroke_width=6)
        )
        # Position cross between col 5 and 6
        cross_pos = (self.grid["C5"] + self.grid["C6"]) / 2
        cross.move_to(cross_pos)

        self.play(FadeIn(person))
        self.play(Create(cross))
        self.wait(2)
