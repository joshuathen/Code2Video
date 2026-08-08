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
        title = "Deconstruction: Components and Magnitude"
        lecture_lines = [
            "A vector decomposes into horizontal and vertical parts.",
            "These components form a right-angled triangle.",
            "Use the Pythagorean theorem to find the vector's length.",
            "This length is what we call the magnitude.",
            "Watch the components change as the vector moves."
        ]
        self.setup_layout(title, lecture_lines)

        # Define Colors
        COLOR_VECTOR = "#FF00FF"  # Magenta
        COLOR_X = "#FFA500"       # Orange
        COLOR_Y = "#ADD8E6"       # Light Blue
        COLOR_TEXT = "#FFFFFF"    # White

        # Setup Plane
        plane = NumberPlane(
            x_range=[-1, 6, 1],
            y_range=[-1, 6, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_numbers": True, "font_size": 20}
        )
        self.place_in_area(plane, 'A1', 'F6', scale_factor=0.9)
        self.add(plane)

        # Value trackers for interactive movement
        target_x = ValueTracker(4)
        target_y = ValueTracker(3)

        # Vector and Components
        # Use simple Vector mobject and update its points to avoid expensive re-creation
        vector = Vector([4, 3], color=COLOR_VECTOR).move_to(plane.c2p(0,0), aligned_edge=DL)
        vector.add_updater(lambda m: m.become(Vector([target_x.get_value(), target_y.get_value()], color=COLOR_VECTOR).move_to(plane.c2p(0,0), aligned_edge=DL)))

        # Components as segments
        x_comp = Line(plane.c2p(0, 0), plane.c2p(4, 0), color=COLOR_X, stroke_width=6)
        x_comp.add_updater(lambda m: m.set_points_as_corners([plane.c2p(0, 0), plane.c2p(target_x.get_value(), 0)]))

        y_comp = Line(plane.c2p(4, 0), plane.c2p(4, 3), color=COLOR_Y, stroke_width=6)
        y_comp.add_updater(lambda m: m.set_points_as_corners([plane.c2p(target_x.get_value(), 0), plane.c2p(target_x.get_value(), target_y.get_value())]))

        # Dashed projection lines
        dashed_x = DashedLine(plane.c2p(4, 3), plane.c2p(4, 0), color=GRAY)
        dashed_x.add_updater(lambda m: m.set_points_as_corners([plane.c2p(target_x.get_value(), target_y.get_value()), plane.c2p(target_x.get_value(), 0)]))
        
        dashed_y = DashedLine(plane.c2p(4, 3), plane.c2p(0, 3), color=GRAY)
        dashed_y.add_updater(lambda m: m.set_points_as_corners([plane.c2p(target_x.get_value(), target_y.get_value()), plane.c2p(0, target_y.get_value())]))

        # Equation Mobjects
        formula = MathTex(r"\sqrt{x^2 + y^2} = \text{length}", color=COLOR_TEXT, font_size=32)
        self.place_at_grid(formula, 'B1', scale_factor=0.8)

        # Note: MathTex re-creation is slightly expensive but limited here to 2 lines
        calc = MathTex(r"\sqrt{4.0^2 + 3.0^2} = 5.0", color=COLOR_TEXT, font_size=28).next_to(formula, DOWN, buff=0.2)
        calc.add_updater(lambda m: m.become(MathTex(
            rf"\sqrt{{{target_x.get_value():.1f}^2 + {target_y.get_value():.1f}^2}} = {np.sqrt(target_x.get_value()**2 + target_y.get_value()**2):.1f}",
            color=COLOR_TEXT, font_size=28
        ).next_to(formula, DOWN, buff=0.2)))

        mag_text = MathTex(r"\text{Magnitude} = 5.00", color=COLOR_VECTOR, font_size=30).next_to(calc, DOWN, buff=0.3)
        mag_text.add_updater(lambda m: m.become(MathTex(
            rf"\text{{Magnitude}} = {np.sqrt(target_x.get_value()**2 + target_y.get_value()**2):.2f}",
            color=COLOR_VECTOR, font_size=30
        ).next_to(calc, DOWN, buff=0.3)))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(vector), run_time=1.5)
        self.play(Create(dashed_x), Create(dashed_y), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Create(x_comp), Create(y_comp), run_time=1.5)
        
        # Right angle
        ra = RightAngle(Line(LEFT, ORIGIN), Line(ORIGIN, UP), length=0.2, color=WHITE)
        ra.add_updater(lambda m: m.move_to(plane.c2p(target_x.get_value(), 0), anchored_edge=UR))
        self.add(ra)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(Write(formula))
        self.play(FadeIn(calc))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(Write(mag_text))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        self.play(target_x.animate.set_value(3), target_y.animate.set_value(4), run_time=2)
        self.wait(0.5)
        self.play(target_x.animate.set_value(5), target_y.animate.set_value(2), run_time=2)
        self.wait(2)
        self.lecture[4].set_color(WHITE)
