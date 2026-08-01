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
        # Topic: The Bridge Between Change and Accumulation: The Relationship Between Integrals and Derivatives
        # Section 6: Conclusion: The Power of the Relationship
        
        title_text = "Conclusion: The Power of the Relationship"
        lecture_lines = [
            "Switching between rates and totals solves complex problems.",
            "Calculus powers physics, engineering, and even economics.",
            "Derivatives and integrals are two halves of one whole."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Turbo the Cheetah [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg] (#FFD700) crosses a checkered finish line.
        color_turbo = "#FFD700"
        self.play(self.lecture[0].animate.set_color(color_turbo))
        
        # Create finish line (checkered pattern)
        sq_size = 0.2
        finish_line = VGroup()
        for r in range(5):
            for c in range(2):
                sq_color = WHITE if (r + c) % 2 == 0 else BLACK
                sq = Square(side_length=sq_size, fill_opacity=1, color=sq_color, stroke_width=0.2)
                sq.move_to(np.array([c * sq_size, -r * sq_size, 0]))
                finish_line.add(sq)
        
        # Issue 43 Fix: Position finish line at C5
        self.place_at_grid(finish_line, "C5")
        
        # Issue 29 Fix: Use Asset for Turbo
        # Issue 42 Fix: Place turbo at C2
        turbo = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg").set_color(color_turbo)
        self.place_at_grid(turbo, "C2", scale_factor=0.6)
        
        self.add(finish_line)
        self.play(Create(turbo))
        # Linear movement to suggest constant speed/crossing finish line at C5
        self.play(turbo.animate.move_to(self.grid["C5"] + RIGHT*0.6), run_time=1.5, rate_func=linear)
        self.wait(0.5)
        self.play(FadeOut(turbo), FadeOut(finish_line))

        # === Animation for Lecture Line 2 ===
        # Icons for a battery (#32CD32) and rocket (#FF8C00) pop up briefly.
        color_physics = "#32CD32"
        self.play(self.lecture[1].animate.set_color(color_physics))
        
        # Battery Icon
        battery_body = Rectangle(width=0.4, height=0.6, color="#32CD32", fill_opacity=0.8)
        battery_tip = Rectangle(width=0.15, height=0.1, color="#32CD32", fill_opacity=1).next_to(battery_body, UP, buff=0.05)
        battery_charge = Rectangle(width=0.3, height=0.4, color="#32CD32", fill_opacity=1).move_to(battery_body)
        battery = VGroup(battery_body, battery_tip, battery_charge)
        self.place_in_area(battery, "B1", "C3", scale_factor=1.0)
        
        # Rocket Icon
        rocket_body = Rectangle(width=0.3, height=0.7, color="#FF8C00", fill_opacity=0.8)
        rocket_nose = Triangle(color="#FF8C00", fill_opacity=1).scale(0.15).next_to(rocket_body, UP, buff=0)
        rocket_fins = VGroup(
            Triangle(color="#FF8C00", fill_opacity=1).scale(0.1).rotate(PI).next_to(rocket_body, LEFT+DOWN, buff=0),
            Triangle(color="#FF8C00", fill_opacity=1).scale(0.1).rotate(PI).next_to(rocket_body, RIGHT+DOWN, buff=0)
        )
        rocket = VGroup(rocket_body, rocket_nose, rocket_fins)
        self.place_in_area(rocket, "D4", "F6", scale_factor=1.0)
        
        self.play(
            FadeIn(battery, scale=1.2),
            FadeIn(rocket, scale=1.2),
            run_time=0.8
        )
        self.wait(1.5)
        self.play(FadeOut(battery), FadeOut(rocket))

        # === Animation for Lecture Line 3 ===
        # Text 'Rates <-> Totals' appears in bright white (#FFFFFF) center-screen.
        color_summary = "#FFFFFF"
        self.play(self.lecture[2].animate.set_color(color_summary))
        
        # Issue 41 Fix: Scale factor 1.0, area C1-D6. 
        # Reusable Belief B008: Use Text instead of MathTex for safety and alphanumeric consistency.
        rates_totals = Text("Rates <-> Totals", color=color_summary, font_size=32)
        self.place_in_area(rates_totals, "C1", "D6", scale_factor=1.0)
        
        self.play(Write(rates_totals))
        self.wait(3)
