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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup layout with the section title and lecture lines
        self.setup_layout(
            "Prerequisite Refresh: Slopes and Slices",
            [
                "Derivatives find the slope of a tangent line.",
                "Integrals calculate the area under a curve.",
                "Are these two mathematical operations somehow connected?"
            ]
        )

        # Define colors
        COLOR_DERIVATIVE = "#FF0000"  # Red
        COLOR_INTEGRAL = "#00FFFF"    # Cyan
        COLOR_TANGENT = "#FFFF00"     # Yellow
        
        # Function to use for both visuals
        def func(x):
            return 0.2 * x**2 + 0.5

        # === Animation for Lecture Line 1 ===
        # Visualization: A curve y = f(x) (red) appears with a tangent line slope shown at a single point.
        
        # Left visual container
        axes_diff = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            axis_config={"include_tip": True},
            x_length=3.5,
            y_length=3.5
        )
        # Apply Issue 24 fix: B1 to E3, scale 0.8
        self.place_in_area(axes_diff, "B1", "E3", scale_factor=0.8)
        
        curve_diff = axes_diff.plot(func, x_range=[0, 3.5], color=COLOR_DERIVATIVE)
        
        # Tangent line at x = 2.0
        point_x = 2.0
        # For a curve plotted from x=0 to 3.5, alpha for x=2.0 is 2.0/3.5
        alpha_val = point_x / 3.5
        tangent_line = TangentLine(curve_diff, alpha=alpha_val, length=2.5, color=COLOR_TANGENT)
        dot = Dot(axes_diff.c2p(point_x, func(point_x)), color=WHITE, radius=0.06)
        
        # Highlight lecture line 1
        self.lecture[0].set_color(COLOR_DERIVATIVE)
        
        self.play(
            Create(axes_diff),
            Create(curve_diff),
            run_time=1.5
        )
        self.play(
            FadeIn(dot),
            Create(tangent_line),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visualization: Screen splits: the right side displays the same curve filled with cyan Riemann rectangles.
        
        # Right visual container
        axes_int = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            axis_config={"include_tip": True},
            x_length=3.5,
            y_length=3.5
        )
        # Apply Issue 25 fix: B4 to E6, scale 0.8
        self.place_in_area(axes_int, "B4", "E6", scale_factor=0.8)
        
        curve_int = axes_int.plot(func, x_range=[0, 3.5], color=COLOR_DERIVATIVE)
        
        # Initial coarse Riemann rectangles
        rects = axes_int.get_riemann_rectangles(
            curve_int,
            x_range=[0.5, 3.0],
            dx=0.5,
            color=COLOR_INTEGRAL,
            fill_opacity=0.6,
            stroke_width=1
        )
        
        # Highlight lecture line 2
        self.lecture[1].set_color(COLOR_INTEGRAL)
        
        self.play(
            Create(axes_int),
            Create(curve_int),
            run_time=1.5
        )
        self.play(
            Create(rects),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visualization: The rectangles become infinitely thin, merging into a smooth solid area under the curve.
        
        # Highlight lecture line 3
        self.lecture[2].set_color(WHITE)
        
        # Refined rectangles (Step 1: Increase count)
        rects_fine = axes_int.get_riemann_rectangles(
            curve_int,
            x_range=[0.5, 3.0],
            dx=0.08,
            color=COLOR_INTEGRAL,
            fill_opacity=0.6,
            stroke_width=0.2
        )
        
        # Final solid area
        final_area = axes_int.get_area(
            curve_int,
            x_range=[0.5, 3.0],
            color=COLOR_INTEGRAL,
            opacity=0.8
        )
        
        # Animate the transition from coarse rectangles to fine to solid area
        self.play(
            Transform(rects, rects_fine),
            run_time=2
        )
        self.play(
            Transform(rects, final_area),
            run_time=1
        )
        
        self.wait(2)
