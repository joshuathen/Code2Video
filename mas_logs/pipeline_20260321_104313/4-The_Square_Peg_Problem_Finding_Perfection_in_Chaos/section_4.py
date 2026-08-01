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

class Section4Scene(TeachingScene):
    def construct(self):
        # 1. Initialize Layout
        # Call to setup_layout with specified title and lecture lines
        self.setup_layout("The Strategy: Mapping Pairs of Points", [
            "Instead of squares, let's track pairs of points.",
            "Every pair is a potential side or diagonal.",
            "We plot every possible pair in a new space."
        ])

        # === Animation for Lecture Line 1 ===
        # Description: A closed curve in #FFFFFF has two dots, A (#FF0000) and B (#0000FF), appearing on it.
        
        jordan_curve = ParametricFunction(
            lambda t: np.array([
                1.5 * np.cos(t) + 0.1 * np.cos(3 * t),
                1.5 * np.sin(t) + 0.2 * np.sin(2 * t),
                0
            ]),
            t_range=[0, TAU],
            color=WHITE
        )
        
        # Issue 47 & 49: Move jordan_curve to area A1-C3 and scale 0.7 to reserve D1-F6
        self.place_in_area(jordan_curve, 'A1', 'C3', scale_factor=0.7)
        
        # Dot A and Label A
        dot_a = Dot(jordan_curve.point_from_proportion(0.1), color="#FF0000", radius=0.1)
        label_a = Text("A", font_size=20, color="#FF0000")
        label_a.next_to(dot_a, UP, buff=0.1)
        
        # Dot B and Label B
        dot_b = Dot(jordan_curve.point_from_proportion(0.4), color="#0000FF", radius=0.1)
        label_b = Text("B", font_size=20, color="#0000FF")
        label_b.next_to(dot_b, DOWN, buff=0.1)

        # Highlight lecture line and animate elements
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        self.play(Create(jordan_curve))
        self.play(FadeIn(dot_a), FadeIn(dot_b), Write(label_a), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Description: A dashed white line connects A and B, representing a potential side or diagonal.
        
        connecting_line = DashedLine(dot_a.get_center(), dot_b.get_center(), color=WHITE)
        
        # Animation Line 2 highlight and creation
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(Create(connecting_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: Dots A and B move around the curve, and the connecting line changes length and orientation.
        
        # Issue 48: highlight_circle at B2, scale 0.5
        highlight_circle = Circle(radius=0.2, color=GREEN)
        self.place_at_grid(highlight_circle, "B2", scale_factor=0.5)

        # Highlight lecture line 3
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        # Use Updaters to keep elements connected during motion
        def update_line(l):
            l.become(DashedLine(dot_a.get_center(), dot_b.get_center(), color=WHITE))
            
        connecting_line.add_updater(update_line)
        label_a.add_updater(lambda l: l.next_to(dot_a, UP, buff=0.1))
        label_b.add_updater(lambda l: l.next_to(dot_b, DOWN, buff=0.1))

        # Track proportions along the curve for dot movement
        alpha = ValueTracker(0.1)
        beta = ValueTracker(0.4)
        
        dot_a.add_updater(lambda d: d.move_to(jordan_curve.point_from_proportion(alpha.get_value() % 1)))
        dot_b.add_updater(lambda d: d.move_to(jordan_curve.point_from_proportion(beta.get_value() % 1)))
        
        # Simultaneous movement of dots and creation of focus highlight
        self.play(
            alpha.animate.set_value(0.6),
            beta.animate.set_value(0.9),
            Create(highlight_circle),
            run_time=4,
            rate_func=linear
        )
        self.play(Indicate(highlight_circle))
        self.wait(2)

        # Clean up updaters
        connecting_line.clear_updaters()
        dot_a.clear_updaters()
        dot_b.clear_updaters()
        label_a.clear_updaters()
        label_b.clear_updaters()
