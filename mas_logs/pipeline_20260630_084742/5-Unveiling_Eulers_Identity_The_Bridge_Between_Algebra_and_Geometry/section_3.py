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
        # Setup context with updated lecture lines from prompt
        title = "The Twist: Imaginary Growth as Rotation"
        # Removed LaTeX delimiters ($) as the base class uses Text() which does not support LaTeX
        lines = [
            'Multiplying growth by i creates a fundamental change.',
            'Now, the growth force acts at a right angle.',
            'This constant sideways push creates a rotation.',
            'Instead of flying away, the point travels in circles.',
            'This continuous turn defines the unit circle.'
        ]
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Fixed FileNotFoundError: Replaced MathTex with Text to avoid LaTeX system dependency
        formula = Text("e^ix", font_size=48, color=WHITE)
        self.place_at_grid(formula, "A4", scale_factor=1.0)
        
        # Unit Circle setup
        radius = 1.5
        center_pos = self.grid["D4"]
        unit_circle = Circle(radius=radius, color="#555555", stroke_width=2)
        self.place_at_grid(unit_circle, "D4")
        
        self.play(
            FadeIn(formula),
            Create(unit_circle),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        theta = ValueTracker(0)
        
        # Growth vector (radial) - Green #00FF00
        growth_vec = Arrow(
            start=center_pos,
            end=center_pos + radius * np.array([1, 0, 0]),
            color="#00FF00",
            buff=0,
            tip_length=0.2
        )
        
        # Perpendicular vector (tangent) - Cyan #00FFFF
        perp_vec = Arrow(
            start=center_pos + radius * np.array([1, 0, 0]),
            end=center_pos + radius * np.array([1, 0, 0]) + np.array([0, 1.0, 0]),
            color="#00FFFF",
            buff=0,
            tip_length=0.2
        )

        # Efficient updaters using put_start_and_end_on
        def update_growth_vec(m):
            t = theta.get_value()
            m.put_start_and_end_on(
                center_pos,
                center_pos + radius * np.array([np.cos(t), np.sin(t), 0])
            )

        def update_perp_vec(m):
            t = theta.get_value()
            tip_pos = center_pos + radius * np.array([np.cos(t), np.sin(t), 0])
            tangent_dir = np.array([-np.sin(t), np.cos(t), 0])
            m.put_start_and_end_on(
                tip_pos,
                tip_pos + tangent_dir * 1.0
            )

        self.play(Create(growth_vec), run_time=1)
        self.play(Create(perp_vec), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Active growth/rotation logic
        growth_vec.add_updater(update_growth_vec)
        perp_vec.add_updater(update_perp_vec)
        
        self.play(theta.animate.set_value(PI/2), run_time=2, rate_func=smooth)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Trace path using a tracking dot
        dot = Dot(color=YELLOW).scale(0.01)
        dot.add_updater(lambda m: m.move_to(center_pos + radius * np.array([np.cos(theta.get_value()), np.sin(theta.get_value()), 0])))
        self.add(dot)
        
        path = TracedPath(dot.get_center, stroke_width=4, color="#FFFF00")
        self.add(path)
        
        # Complete the rotation
        current_theta = theta.get_value()
        self.play(theta.animate.set_value(current_theta + 2*PI), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Highlight the finished circle path
        highlight_circle = Circle(radius=radius, color="#FFFF00", stroke_width=4)
        self.place_at_grid(highlight_circle, "D4")
        
        self.play(
            Create(highlight_circle),
            formula.animate.set_color(YELLOW),
            run_time=1.5
        )
        self.wait(2)
