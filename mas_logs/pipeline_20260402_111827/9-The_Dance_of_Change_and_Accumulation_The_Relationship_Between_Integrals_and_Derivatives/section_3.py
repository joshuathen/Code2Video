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
        # Define the lecture lines for layout
        lecture_lines = [
            "Velocity graphs show how speed changes.",
            "We approximate the area using simple blocks.",
            "Thinner slices provide a more accurate total.",
            "The integral represents this total shaded area.",
            "Area under velocity gives the total distance."
        ]
        
        # Initialize layout
        self.setup_layout("The Integral: Shading the Accumulation", lecture_lines)

        # Color palette
        COLOR_CURVE = "#00FFFF"  # Cyan
        COLOR_BARS = "#FFD700"   # Yellow
        COLOR_AREA = "#FFA500"   # Orange

        # Setup Axes and Graph elements
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 4, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE},
            tips=False
        )
        # Issue 33: Scale reduced from 0.9 to 0.8
        self.place_in_area(axes, "A1", "F6", scale_factor=0.8)

        # Velocity function: v(t) = 0.5*sin(1.5t) + 2
        def v_func(t):
            return 0.5 * np.sin(1.5 * t) + 2.0

        curve = axes.plot(v_func, x_range=[0, 4.5], color=COLOR_CURVE)
        curve_label = Text("v(t)", color=COLOR_CURVE, font_size=24)
        # Issue 34: Position moved from B6 to B5 and scale reduced to 0.8
        self.place_at_grid(curve_label, "B5", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Description: A wavy velocity curve v(t) is drawn in cyan (#00FFFF) on the screen.
        self.play(self.lecture[0].animate.set_color(COLOR_CURVE))
        self.play(
            Write(axes),
            Create(curve),
            Write(curve_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Description: Five thick vertical bars (#FFD700) fill the space between the curve and the x-axis.
        self.play(self.lecture[1].animate.set_color(COLOR_BARS))
        rects_5 = axes.get_riemann_rectangles(
            curve, 
            x_range=[0.5, 4.5], 
            dx=0.8, 
            color=COLOR_BARS, 
            fill_opacity=0.5, 
            stroke_width=2
        )
        self.play(Create(rects_5), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: The bars multiply into 50 thin slices, fitting the curve's shape precisely.
        self.play(self.lecture[2].animate.set_color(COLOR_BARS))
        rects_50 = axes.get_riemann_rectangles(
            curve, 
            x_range=[0.5, 4.5], 
            dx=0.08, 
            color=COLOR_BARS, 
            fill_opacity=0.5, 
            stroke_width=0.1
        )
        self.play(Transform(rects_5, rects_50), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Description: The slices blend into a semi-transparent orange (#FFA500) shaded area.
        self.play(self.lecture[3].animate.set_color(COLOR_AREA))
        
        integral_area = axes.get_area(
            curve, 
            x_range=[0.5, 4.5], 
            color=COLOR_AREA, 
            opacity=0.4
        )
        
        self.play(
            ReplacementTransform(rects_5, integral_area),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Description: The label 'Accumulated Distance (Integral)' appears over the shaded region.
        self.play(self.lecture[4].animate.set_color(COLOR_AREA))
        
        integral_label = Text("Accumulated Distance (Integral)", font_size=24, color=COLOR_AREA)
        # Issue 32: Positioning changed from B3 to area A2-A5, scale 0.6
        self.place_in_area(integral_label, "A2", "A5", scale_factor=0.6)

        self.play(
            Write(integral_label),
            run_time=1.5
        )
        self.wait(3)
