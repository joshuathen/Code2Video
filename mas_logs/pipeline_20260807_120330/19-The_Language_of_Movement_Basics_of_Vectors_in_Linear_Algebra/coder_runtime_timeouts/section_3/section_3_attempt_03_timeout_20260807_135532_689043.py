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
        # --- Data setup ---
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
        COLOR_VEC = "#FF00FF"  # Magenta
        COLOR_X = "#FFA500"    # Orange
        COLOR_Y = "#ADD8E6"    # Light Blue
        COLOR_EQ = "#FFFFFF"   # White
        COLOR_HL = "#FFFF00"   # Yellow highlight

        # Coordinate System - Optimized for performance
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 5, 1],
            x_length=4.0,
            y_length=3.0,
            axis_config={"include_tip": False, "font_size": 16, "stroke_width": 2},
        ).add_coordinates()
        self.place_in_area(axes, 'C2', 'F6')
        self.add(axes)

        # Trackers
        tx = ValueTracker(4)
        ty = ValueTracker(3)

        # Vector and Components - Using updaters for smooth motion
        # Persistent objects to avoid re-creation
        vector = Arrow(
            start=axes.c2p(0, 0),
            end=axes.c2p(4, 3),
            buff=0,
            color=COLOR_VEC,
            stroke_width=5
        )
        vector.add_updater(lambda m: m.put_start_and_end_on(axes.c2p(0, 0), axes.c2p(tx.get_value(), ty.get_value())))

        x_line = Line(color=COLOR_X, stroke_width=5)
        x_line.add_updater(lambda m: m.set_points_as_corners([axes.c2p(0, 0), axes.c2p(tx.get_value(), 0)]))

        y_line = Line(color=COLOR_Y, stroke_width=5)
        y_line.add_updater(lambda m: m.set_points_as_corners([axes.c2p(tx.get_value(), 0), axes.c2p(tx.get_value(), ty.get_value())]))

        dashed_x = DashedLine(dash_length=0.1, color=GRAY_A).add_updater(
            lambda m: m.set_points_as_corners([axes.c2p(tx.get_value(), ty.get_value()), axes.c2p(tx.get_value(), 0)])
        )
        dashed_y = DashedLine(dash_length=0.1, color=GRAY_A).add_updater(
            lambda m: m.set_points_as_corners([axes.c2p(tx.get_value(), ty.get_value()), axes.c2p(0, ty.get_value())])
        )

        # Equation and Labels - Positioned to avoid overlap
        eq_base = MathTex(r"\sqrt{4^2 + 3^2} = 5", color=COLOR_EQ, font_size=32)
        self.place_at_grid(eq_base, 'B2', scale_factor=1.0)
        
        mag_label = Text("Magnitude =", font_size=24, color=COLOR_VEC)
        mag_val = DecimalNumber(5.0, num_decimal_places=2, color=COLOR_VEC, font_size=24)
        mag_grp = VGroup(mag_label, mag_val).arrange(RIGHT, buff=0.2)
        self.place_at_grid(mag_grp, 'B5', scale_factor=1.0)
        mag_val.add_updater(lambda d: d.set_value(np.sqrt(tx.get_value()**2 + ty.get_value()**2)))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_HL)
        self.play(Create(vector), run_time=0.8)
        self.play(Create(dashed_x), Create(dashed_y), run_time=0.8)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(COLOR_VEC)
        self.lecture[1].set_color(COLOR_HL)
        self.play(Create(x_line), Create(y_line), run_time=0.8)
        
        # Simple right angle box
        ra_box = Square(side_length=0.2, stroke_width=2, color=WHITE)
        ra_box.add_updater(lambda m: m.move_to(axes.c2p(tx.get_value(), 0), anchored_edge=UR))
        self.add(ra_box)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(COLOR_Y)
        self.lecture[2].set_color(COLOR_HL)
        self.play(Write(eq_base), run_time=0.8)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(COLOR_EQ)
        self.lecture[3].set_color(COLOR_HL)
        self.play(FadeIn(mag_grp), run_time=0.8)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(COLOR_VEC)
        self.lecture[4].set_color(COLOR_HL)
        # Dynamic movement demonstration
        self.play(tx.animate.set_value(3), ty.animate.set_value(4), run_time=1.5, rate_func=linear)
        self.play(tx.animate.set_value(5), ty.animate.set_value(2), run_time=1.5, rate_func=linear)
        self.wait(1)

        # Final cleanup - restore base color
        self.lecture[4].set_color(COLOR_EQ)
        self.wait(2)
