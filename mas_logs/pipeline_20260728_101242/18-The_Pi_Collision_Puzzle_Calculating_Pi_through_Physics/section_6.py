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
        title = "The Grand Conclusion: Why Pi?"
        lines = [
            "Large mass ratios result in very small bounce angles.",
            "Each collision covers a tiny arc of the circle.",
            "Total collisions equal the total arc divided by step.",
            "This ratio directly relates to the circle's circumference.",
            "Pi appears because we are traversing a geometric circle."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_CIRCLE = "#FFFF00"  # Yellow
        COLOR_THETA = "#FF4D4F"   # Red
        COLOR_FORMULA = "#FFFFFF" # White

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Representing a small portion of a huge circle.
        main_arc = Arc(radius=10, start_angle=PI/2 - 0.2, angle=0.4, color=COLOR_CIRCLE)
        # Apply fix from Issue 33: Position arc in a more efficient area
        self.place_in_area(main_arc, 'A2', 'F5', scale_factor=0.9)
        
        self.play(Create(main_arc), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_THETA)
        
        # Define two points on the arc to represent a "step"
        point_start = main_arc.point_from_proportion(0.4)
        point_end = main_arc.point_from_proportion(0.6)
        
        # Collision arc follow the main arc curvature
        collision_arc = ArcBetweenPoints(point_start, point_end, radius=10, color=COLOR_THETA)
        collision_arc.set_stroke(width=6)
        
        dot_start = Dot(point_start, color=COLOR_THETA, radius=0.06)
        dot_end = Dot(point_end, color=COLOR_THETA, radius=0.06)

        self.play(Create(collision_arc), FadeIn(dot_start), FadeIn(dot_end))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_FORMULA)
        
        # Label theta - apply fix from Issue 35: Align label horizontally
        theta_label = MathTex(r"\theta", color=COLOR_THETA)
        self.place_in_area(theta_label, 'B3', 'B4', scale_factor=0.8)
        
        # Formula theta approx sqrt(m/M)
        theta_formula = MathTex(r"\theta \approx \sqrt{\frac{m}{M}}", color=COLOR_FORMULA)
        self.place_in_area(theta_formula, "D2", "D5", scale_factor=0.9)
        
        self.play(Write(theta_label))
        self.play(Write(theta_formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_FORMULA)
        
        # Formula N = pi / theta
        n_formula = MathTex(r"N = \frac{\pi}{\theta}", color=COLOR_FORMULA)
        self.place_in_area(n_formula, "E2", "E5", scale_factor=1.0)
        
        self.play(Write(n_formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_THETA)
        
        # Big Pi symbol - apply fix from Issue 34: Center symbol relative to formulas
        pi_symbol = MathTex(r"\pi", color=COLOR_THETA).scale(3)
        self.place_in_area(pi_symbol, 'C3', 'C4', scale_factor=1.2)
        
        # Create a glow effect
        pi_glow = pi_symbol.copy().scale(1.2).set_opacity(0.3)
        
        self.play(
            FadeIn(pi_symbol),
            pi_symbol.animate.scale(1.5),
            Flash(pi_symbol, color=COLOR_THETA)
        )
        self.play(
            FadeIn(pi_glow),
            pi_glow.animate.scale(1.2).set_opacity(0)
        )
        
        self.wait(3)

        # Final Cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(2)
