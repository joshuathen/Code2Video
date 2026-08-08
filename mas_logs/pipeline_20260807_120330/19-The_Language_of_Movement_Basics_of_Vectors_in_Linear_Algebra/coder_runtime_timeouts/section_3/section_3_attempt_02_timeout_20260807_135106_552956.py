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

        # Colors
        COLOR_VECTOR = "#FF00FF"  # Magenta
        COLOR_X = "#FFA500"       # Orange
        COLOR_Y = "#ADD8E6"       # Light Blue
        COLOR_HIGHLIGHT = "#FFFF00" # Yellow for active line
        COLOR_TEXT = "#FFFFFF"

        # Apply matching colors to lecture lines initially
        self.lecture[0].set_color(COLOR_VECTOR)
        self.lecture[1].set_color(COLOR_Y)
        self.lecture[2].set_color(COLOR_TEXT)
        self.lecture[3].set_color(COLOR_VECTOR)
        self.lecture[4].set_color(COLOR_TEXT)

        # Setup Plane - optimized sizing for right-side area
        plane = NumberPlane(
            x_range=[-1, 6, 1],
            y_range=[-1, 6, 1],
            x_length=4.5,
            y_length=4.5,
            background_line_style={"stroke_opacity": 0.3},
            axis_config={"include_numbers": True, "font_size": 18}
        )
        # Center the plane in the bottom-right grid area
        self.place_in_area(plane, 'B1', 'F6', scale_factor=1.0)
        self.add(plane)

        # Value trackers for coordinates
        target_x = ValueTracker(4)
        target_y = ValueTracker(3)

        # Vector - using put_start_and_end_on for efficiency
        vector = Arrow(plane.c2p(0, 0), plane.c2p(4, 3), buff=0, color=COLOR_VECTOR, stroke_width=4)
        vector.add_updater(lambda m: m.put_start_and_end_on(
            plane.c2p(0, 0), 
            plane.c2p(target_x.get_value(), target_y.get_value())
        ))

        # Components
        x_comp = Line(plane.c2p(0, 0), plane.c2p(4, 0), color=COLOR_X, stroke_width=6)
        x_comp.add_updater(lambda m: m.set_points_as_corners([
            plane.c2p(0, 0), 
            plane.c2p(target_x.get_value(), 0)
        ]))

        y_comp = Line(plane.c2p(4, 0), plane.c2p(4, 3), color=COLOR_Y, stroke_width=6)
        y_comp.add_updater(lambda m: m.set_points_as_corners([
            plane.c2p(target_x.get_value(), 0), 
            plane.c2p(target_x.get_value(), target_y.get_value())
        ]))

        # Projection lines
        dashed_x = DashedLine(plane.c2p(4, 3), plane.c2p(4, 0), color=GRAY_B)
        dashed_x.add_updater(lambda m: m.set_points_as_corners([
            plane.c2p(target_x.get_value(), target_y.get_value()), 
            plane.c2p(target_x.get_value(), 0)
        ]))

        dashed_y = DashedLine(plane.c2p(4, 3), plane.c2p(0, 3), color=GRAY_B)
        dashed_y.add_updater(lambda m: m.set_points_as_corners([
            plane.c2p(target_x.get_value(), target_y.get_value()), 
            plane.c2p(0, target_y.get_value())
        ]))

        # Equations - Pre-built static parts to optimize render
        formula = MathTex(r"\sqrt{x^2 + y^2} = L", color=COLOR_TEXT, font_size=32)
        self.place_at_grid(formula, 'B2', scale_factor=0.9)

        # Dynamic calculation group using DecimalNumber (much faster than MathTex updaters)
        calc_vgroup = VGroup()
        sqrt_sym = MathTex(r"\sqrt{", font_size=28)
        x_val = DecimalNumber(4.0, num_decimal_places=1, font_size=28)
        plus_sym = MathTex(r"^2 + ", font_size=28)
        y_val = DecimalNumber(3.0, num_decimal_places=1, font_size=28)
        equal_sym = MathTex(r"^2} = ", font_size=28)
        res_val = DecimalNumber(5.0, num_decimal_places=1, font_size=28)
        calc_vgroup.add(sqrt_sym, x_val, plus_sym, y_val, equal_sym, res_val).arrange(RIGHT, buff=0.1)
        self.place_at_grid(calc_vgroup, 'C2', scale_factor=0.9)

        # Connect DecimalNumbers to trackers
        x_val.add_updater(lambda d: d.set_value(target_x.get_value()))
        y_val.add_updater(lambda d: d.set_value(target_y.get_value()))
        res_val.add_updater(lambda d: d.set_value(np.sqrt(target_x.get_value()**2 + target_y.get_value()**2)))
        calc_vgroup.add_updater(lambda m: m.arrange(RIGHT, buff=0.1).move_to(self.grid['C2']))

        # Magnitude display
        mag_label = Text("Magnitude: ", font_size=24, color=COLOR_VECTOR)
        mag_num = DecimalNumber(5.0, num_decimal_places=2, font_size=24, color=COLOR_VECTOR)
        mag_group = VGroup(mag_label, mag_num).arrange(RIGHT, buff=0.1)
        self.place_at_grid(mag_group, 'D2', scale_factor=0.9)
        mag_num.add_updater(lambda d: d.set_value(np.sqrt(target_x.get_value()**2 + target_y.get_value()**2)))
        mag_group.add_updater(lambda m: m.arrange(RIGHT, buff=0.1).move_to(self.grid['D2']))

        # === Animation for Lecture Line 1 ===
        # Highlight active line
        prev_color_1 = self.lecture[0].color
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        self.play(Create(vector), run_time=1)
        self.play(Create(dashed_x), Create(dashed_y), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(prev_color_1)
        prev_color_2 = self.lecture[1].color
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        self.play(Create(x_comp), Create(y_comp), run_time=1)
        
        # Right angle symbol at the component junction
        ra = RightAngle(Line(ORIGIN, LEFT), Line(ORIGIN, UP), length=0.2, color=WHITE)
        ra.add_updater(lambda m: m.move_to(plane.c2p(target_x.get_value(), 0), anchored_edge=UR))
        self.add(ra)
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(prev_color_2)
        prev_color_3 = self.lecture[2].color
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        self.play(Write(formula), FadeIn(calc_vgroup))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(prev_color_3)
        prev_color_4 = self.lecture[3].color
        self.lecture[3].set_color(COLOR_HIGHLIGHT)
        self.play(FadeIn(mag_group))
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(prev_color_4)
        prev_color_5 = self.lecture[4].color
        self.lecture[4].set_color(COLOR_HIGHLIGHT)
        
        # Animate changing components to demonstrate the dynamic relationship
        self.play(target_x.animate.set_value(3), target_y.animate.set_value(4), run_time=2, rate_func=linear)
        self.wait(0.5)
        self.play(target_x.animate.set_value(5), target_y.animate.set_value(2), run_time=2, rate_func=linear)
        self.wait(1)
        
        self.lecture[4].set_color(prev_color_5)
        self.wait(2)
