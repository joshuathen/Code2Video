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

class Section5Scene(TeachingScene):
    def construct(self):
        title = "Summary: Independence vs. Bayesian Updating"
        lines = [
            "Independence requires no updates to our knowledge.",
            "Bayes' Theorem provides the math for learning from data.",
            "Evidence always shrinks the space of possibilities."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Color match: Blue (#0000FF)
        self.play(self.lecture[0].animate.set_color("#0000FF"))
        
        blue_line_1 = Line(LEFT * 1.5, RIGHT * 1.5, color="#0000FF", stroke_width=6)
        blue_line_2 = Line(LEFT * 1.5, RIGHT * 1.5, color="#0000FF", stroke_width=6)
        
        # Applying Fix for Issue 29: Use place_in_area for better balance
        self.place_in_area(blue_line_1, 'A2', 'A5')
        self.place_in_area(blue_line_2, 'B2', 'B5')
        
        # Capture starting centers for updaters to maintain positioning during movement
        center_a2_a5 = (self.grid["A2"] + self.grid["A5"]) / 2
        center_b2_b5 = (self.grid["B2"] + self.grid["B5"]) / 2
        
        vt_lines = ValueTracker(0)
        blue_line_1.add_updater(lambda m: m.move_to(center_a2_a5 + RIGHT * vt_lines.get_value()))
        blue_line_2.add_updater(lambda m: m.move_to(center_b2_b5 + RIGHT * vt_lines.get_value()))
        
        self.play(Create(blue_line_1), Create(blue_line_2))
        self.play(vt_lines.animate.set_value(0.5), run_time=2, rate_func=linear)
        
        blue_line_1.clear_updaters()
        blue_line_2.clear_updaters()
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Color match: White (#FFFFFF)
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        
        circle_a = Circle(radius=0.7, color=BLUE, fill_opacity=0.3)
        circle_b = Circle(radius=0.7, color=RED, fill_opacity=0.3)
        circles = VGroup(circle_a, circle_b).arrange(RIGHT, buff=-0.6)
        self.place_in_area(circles, "C2", "D5")
        
        mag_circle = Circle(radius=0.25, color="#FFFFFF", stroke_width=4)
        mag_handle = Line(ORIGIN, DOWN * 0.4, color="#FFFFFF", stroke_width=4).next_to(mag_circle, DOWN, buff=0)
        magnifying_glass = VGroup(mag_circle, mag_handle).rotate(-45 * DEGREES)
        
        # Applying Fix for Issue 30: Move to C2 and scale 0.5 to avoid obstruction
        self.place_at_grid(magnifying_glass, 'C2', scale_factor=0.5)
        
        self.play(FadeIn(circles))
        self.play(FadeIn(magnifying_glass))
        
        # Zoom in on the intersection (scale back up to original intended size)
        self.play(
            magnifying_glass.animate.move_to(circles.get_center()).scale(3.0),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color match: Yellow (#FFFF00)
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        text_indep = Text("Independence", font_size=20, color=WHITE)
        text_vs = Text("vs.", font_size=18, color=WHITE)
        text_bayesian = Text("Bayesian Update", font_size=20, color=WHITE)
        comparison = VGroup(text_indep, text_vs, text_bayesian).arrange(RIGHT, buff=0.3)
        self.place_in_area(comparison, "E1", "E6")
        
        update_world = Text("Update Your World", font_size=32, color="#FFFF00")
        self.place_in_area(update_world, "F1", "F6")
        
        self.play(Write(comparison))
        self.wait(0.5)
        self.play(FadeIn(update_world))
        self.play(Flash(update_world, color="#FFFF00", line_length=0.3, flash_radius=1.5, num_lines=12))
        
        self.wait(3)
