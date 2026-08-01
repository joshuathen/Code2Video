from manim import *
import numpy as np
import random

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
        lecture_lines = [
            "This mathematical vastness protects your bank and Bitcoin.",
            "SHA-256 is the gold standard of modern security.",
            "Your data is safe because the math is massive."
        ]
        self.setup_layout("Conclusion: Security in Scale", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create Solar System
        sun = Dot(color=YELLOW, radius=0.15)
        orbit1 = Circle(radius=0.4, color=GRAY_A, stroke_width=1)
        orbit2 = Circle(radius=0.7, color=GRAY_A, stroke_width=1)
        planet1 = Dot(color=BLUE_C, radius=0.04).move_to(orbit1.point_at_angle(PI/3))
        planet2 = Dot(color=ORANGE, radius=0.05).move_to(orbit2.point_at_angle(5*PI/4))
        solar_system = VGroup(sun, orbit1, orbit2, planet1, planet2)
        
        # Fix Issue 45: scale_factor=1.0
        self.place_in_area(solar_system, "A1", "F6", scale_factor=1.0)
        self.play(Create(solar_system))
        self.wait(1)
        
        # Star field background
        random.seed(42)
        tl = self.grid["A1"]
        br = self.grid["F6"]
        stars = VGroup(*[
            Dot(
                point=np.array([
                    random.uniform(tl[0] - 1.0, br[0] + 1.0),
                    random.uniform(br[1] - 1.0, tl[1] + 1.0),
                    0
                ]),
                radius=0.015,
                color=WHITE,
                fill_opacity=random.uniform(0.3, 0.8)
            ) for _ in range(80)
        ])
        
        # Zoom out simulation
        self.play(
            solar_system.animate.scale(0.15).set_opacity(0.4).move_to(self.grid["C3"]),
            FadeIn(stars),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(LIGHT_PINK))
        
        # Smartphone icon
        phone_color = "#ADD8E6"
        phone_body = RoundedRectangle(corner_radius=0.1, height=1.5, width=0.8, color=phone_color, stroke_width=4)
        phone_screen = Rectangle(height=1.2, width=0.7, color=phone_color, stroke_width=2).move_to(phone_body.get_center())
        phone_home = Circle(radius=0.05, color=phone_color, stroke_width=2).next_to(phone_screen, DOWN, buff=0.05)
        phone_icon = VGroup(phone_body, phone_screen, phone_home)
        
        # Asset: bank.svg (Issue 30 & 52)
        bank_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bank.svg")
        bank_asset.set_color(phone_color).scale(0.3).move_to(phone_screen.get_center())
        
        security_group = VGroup(phone_icon, bank_asset)
        
        # Glow
        glow = Arc(radius=1.2, angle=TAU, color=phone_color, stroke_width=0, fill_opacity=0.1).move_to(security_group.get_center())
        
        # Fix Issue 46: place_in_area('C2', 'C5', scale_factor=0.8)
        self.place_in_area(security_group, "C2", "C5", scale_factor=0.8)
        glow.move_to(security_group.get_center())
        
        self.play(
            FadeIn(security_group),
            FadeIn(glow),
            stars.animate.set_opacity(0.2)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        conclusion_text = Text("Security in Mathematical Scale", font_size=24, color=WHITE)
        # Fix Issue 47: place_in_area('E2', 'E5', scale_factor=1.0)
        self.place_in_area(conclusion_text, "E2", "E5", scale_factor=1.0)
        
        self.play(FadeIn(conclusion_text, shift=UP*0.2))
        self.wait(3)
