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
        # Setup layout with specified lecture lines
        lecture_lines = [
            'Zooming in, we define the points x and x plus h.',
            'The horizontal distance between these points is h.',
            'Now, we slide point B toward point A.',
            'The interval h shrinks toward a value of zero.',
            'As B moves, the secant line rotates and pivots.'
        ]
        self.setup_layout("The Limit: Closing the Gap", lecture_lines)

        # Colors
        COLOR_SECANT = "#F1C40F"
        COLOR_BRACKET = "#F39C12"
        COLOR_TEXT = "#FFFFFF"

        # 1. Setup Axes and Graph
        # Issue 35: Reposition axes to A1-E6
        axes = Axes(
            x_range=[0, 2.5, 0.5],
            y_range=[0, 4, 1],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": True, "color": GREY}
        )
        self.place_in_area(axes, 'A1', 'E6', scale_factor=0.8)
        
        def func(x):
            return x**2

        graph = axes.plot(func, color=BLUE, x_range=[0, 2.1])
        
        # Trackers for animation
        h_tracker = ValueTracker(1.0)
        x_a = 1.0
        
        # Point A (Fixed)
        dot_a = Dot(axes.c2p(x_a, func(x_a)), color=WHITE)
        label_x = Text("x", font_size=24, color=COLOR_TEXT)
        self.place_at_grid(label_x, 'F3', scale_factor=0.5)

        # Dynamic Point B
        dot_b = Dot(color=WHITE)
        dot_b.add_updater(lambda d: d.move_to(axes.c2p(x_a + h_tracker.get_value(), func(x_a + h_tracker.get_value()))))
        
        # Issue 33: label_x_plus_h at F5
        label_x_plus_h = Text("x + h", font_size=24, color=COLOR_TEXT)
        self.place_at_grid(label_x_plus_h, 'F5', scale_factor=0.5)

        # Secant Line
        secant_line = Line(color=COLOR_SECANT)
        def update_secant(l):
            p1 = axes.c2p(x_a, func(x_a))
            p2 = axes.c2p(x_a + h_tracker.get_value(), func(x_a + h_tracker.get_value()))
            v = p2 - p1
            if np.linalg.norm(v) < 0.001:
                # Tangent approximation at x=1
                v = np.array([1, 2, 0]) * 0.001
            # Extend the line visually
            ext_start = p1 - v * 2.0 / np.linalg.norm(v)
            ext_end = p1 + v * 4.0 / np.linalg.norm(v)
            l.put_start_and_end_on(ext_start, ext_end)
            
        secant_line.add_updater(update_secant)

        # Bracket for h (Horizontal distance on x-axis)
        h_bracket = BraceBetweenPoints(
            axes.c2p(x_a, 0),
            axes.c2p(x_a + 1.0, 0),
            direction=DOWN,
            color=COLOR_BRACKET,
            buff=0.1
        )
        h_bracket.add_updater(lambda b: b.become(
            BraceBetweenPoints(
                axes.c2p(x_a, 0),
                axes.c2p(x_a + h_tracker.get_value(), 0),
                direction=DOWN,
                color=COLOR_BRACKET,
                buff=0.1
            )
        ))
        
        # Issue 34: label_h at F4
        label_h = Text("h", font_size=24, color=COLOR_BRACKET)
        self.place_at_grid(label_h, 'F4', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        # "Zooming in, we define the points x and x plus h."
        self.play(self.lecture[0].animate.set_color(YELLOW), run_time=0.5)
        self.play(Create(axes), Create(graph))
        self.play(FadeIn(dot_a), FadeIn(dot_b), FadeIn(label_x), FadeIn(label_x_plus_h))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The horizontal distance between these points is h."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            run_time=0.5
        )
        self.play(Create(h_bracket), FadeIn(label_h))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Now, we slide point B toward point A."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW),
            run_time=0.5
        )
        self.play(Create(secant_line))
        self.play(h_tracker.animate.set_value(0.5), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "The interval h shrinks toward a value of zero."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW),
            run_time=0.5
        )
        # Visual shrink of label to match concept
        self.play(
            h_tracker.animate.set_value(0.1),
            label_h.animate.scale(0.8),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "As B moves, the secant line rotates and pivots."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW),
            run_time=0.5
        )
        # Final approach to zero
        self.play(
            h_tracker.animate.set_value(0.01),
            label_h.animate.scale(0.8),
            run_time=2
        )
        self.wait(2)
        
        # Final State Clean
        self.play(self.lecture[4].animate.set_color(WHITE), run_time=0.5)
