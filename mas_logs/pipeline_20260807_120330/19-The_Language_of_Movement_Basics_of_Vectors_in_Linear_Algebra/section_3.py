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

        # Coordinate System - Optimized for performance
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 6, 1],
            x_length=4.2,
            y_length=4.2,
            axis_config={"include_tip": True, "stroke_width": 2, "color": GRAY},
        ).add_coordinates()
        self.place_in_area(axes, 'C2', 'F6')
        self.add(axes)

        # Vector and Components (Initial: 4, 3)
        vec_start = axes.c2p(0, 0)
        vec_end = axes.c2p(4, 3)
        
        vector = Arrow(start=vec_start, end=vec_end, buff=0, color=COLOR_VEC, stroke_width=6)
        
        dashed_x = DashedLine(start=axes.c2p(4, 3), end=axes.c2p(4, 0), color=GRAY_B)
        dashed_y = DashedLine(start=axes.c2p(4, 3), end=axes.c2p(0, 3), color=GRAY_B)
        
        x_line = Line(start=axes.c2p(0, 0), end=axes.c2p(4, 0), color=COLOR_X, stroke_width=6)
        y_line = Line(start=axes.c2p(4, 0), end=axes.c2p(4, 3), color=COLOR_Y, stroke_width=6)
        
        # Right angle box
        ra_box = RightAngle(x_line, y_line, length=0.2, quadrant=(1,1), color=WHITE)

        # Equation and Label
        equation = MathTex(r"\sqrt{4^2 + 3^2} = 5", color=COLOR_EQ, font_size=32)
        # Issue 18: Fix cramping at B2 by using area B2-B3
        self.place_in_area(equation, 'B2', 'B3', scale_factor=1.0)
        
        mag_label = MathTex(r"\|\vec{v}\| = 5", color=COLOR_VEC, font_size=32)
        # Issue 20: Fix gap at B5 by using area B4-B5
        self.place_in_area(mag_label, 'B4', 'B5', scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        # "A vector decomposes into horizontal and vertical parts."
        self.lecture[0].set_color(COLOR_VEC)
        self.play(Create(vector), run_time=1)
        self.play(Create(dashed_x), Create(dashed_y), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "These components form a right-angled triangle."
        self.lecture[0].set_color(WHITE) # Reset color to focus on next line
        self.lecture[1].set_color(COLOR_X)
        self.play(Create(x_line), Create(y_line), run_time=1)
        self.play(FadeIn(ra_box), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Use the Pythagorean theorem to find the vector's length."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_EQ)
        self.play(Write(equation), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "This length is what we call the magnitude."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_VEC)
        self.play(Write(mag_label), run_time=1)
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # "Watch the components change as the vector moves."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Transform to new state (3, 4) to show change
        new_vec_end = axes.c2p(3, 4)
        new_vector = Arrow(start=vec_start, end=new_vec_end, buff=0, color=COLOR_VEC, stroke_width=6)
        new_x_line = Line(start=axes.c2p(0, 0), end=axes.c2p(3, 0), color=COLOR_X, stroke_width=6)
        new_y_line = Line(start=axes.c2p(3, 0), end=axes.c2p(3, 4), color=COLOR_Y, stroke_width=6)
        new_dashed_x = DashedLine(start=axes.c2p(3, 4), end=axes.c2p(3, 0), color=GRAY_B)
        new_dashed_y = DashedLine(start=axes.c2p(3, 4), end=axes.c2p(0, 4), color=GRAY_B)
        new_ra_box = RightAngle(new_x_line, new_y_line, length=0.2, quadrant=(1,1), color=WHITE)
        
        new_equation = MathTex(r"\sqrt{3^2 + 4^2} = 5", color=COLOR_EQ, font_size=32)
        # Issue 19: Fix cramping for new_equation using area B2-B3
        self.place_in_area(new_equation, 'B2', 'B3', scale_factor=1.0)
        
        self.play(
            Transform(vector, new_vector),
            Transform(x_line, new_x_line),
            Transform(y_line, new_y_line),
            Transform(dashed_x, new_dashed_x),
            Transform(dashed_y, new_dashed_y),
            Transform(ra_box, new_ra_box),
            Transform(equation, new_equation),
            run_time=2
        )
        self.wait(2)

        # Reset final color
        self.lecture[4].set_color(WHITE)
        self.wait(2)
