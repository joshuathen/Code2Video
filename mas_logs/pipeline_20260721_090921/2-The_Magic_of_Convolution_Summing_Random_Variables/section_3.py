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
        # Setup title and lecture lines
        title_text = "The 'Sum Line' Constraint"
        lecture_lines = [
            "We want the probability where their sum equals exactly Z.",
            "Pairs (X, Y) must satisfy X plus Y equals Z.",
            "This creates a diagonal line across our joint probability grid.",
            "A sliding 'scanner' line moves to collect these possible pairs.",
            "Each point on the line contributes to the total probability."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_LINE = "#FF4500"
        COLOR_GRID = "#444444"
        COLOR_ACTIVE = YELLOW

        # Create 2D Joint Probability Space Representation
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"include_tip": False, "color": WHITE},
        )
        
        # Use Text instead of MathTex for stability
        x_label = axes.get_x_axis_label(Text("X", font_size=20), edge=RIGHT, direction=RIGHT, buff=0.1)
        y_label = axes.get_y_axis_label(Text("Y", font_size=20), edge=UP, direction=UP, buff=0.1)
        
        joint_points = VGroup()
        for x in range(6):
            for y in range(6):
                dot = Dot(axes.c2p(x, y, 0), color=COLOR_GRID, radius=0.04)
                joint_points.add(dot)
        
        joint_grid = VGroup(axes, x_label, y_label, joint_points)
        # Apply Fix: [VideoCritic][section_3] Line 85: self.place_in_area(joint_grid, 'B2', 'F5', scale_factor=1.0)
        self.place_in_area(joint_grid, "B2", "F5", scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_ACTIVE)
        self.play(FadeIn(joint_grid))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_ACTIVE)
        
        formula = Text("X + Y = Z", font_size=36, color=COLOR_LINE)
        # Apply Fix: [VideoCritic][section_3] Line 98: self.place_in_area(formula, 'A3', 'A5', scale_factor=0.8)
        self.place_in_area(formula, "A3", "A5", scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_ACTIVE)

        z_tracker = ValueTracker(3)
        
        def get_line_points(z):
            # Line x + y = z intersects the region [0,5] x [0,5]
            x_min = max(0, z - 5)
            x_max = min(z, 5)
            if x_min > x_max:
                return None
            return [axes.c2p(x_min, z - x_min, 0), axes.c2p(x_max, z - x_max, 0)]

        # Initial sum line
        pts = get_line_points(3)
        sum_line = Line(pts[0], pts[1], color=COLOR_LINE, stroke_width=6)
        
        # Load Scanner Icon Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/scanner.svg]
        scanner_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scanner.svg")
        scanner_icon.set_color(COLOR_LINE).scale(0.3)
        scanner_icon.move_to(sum_line.get_center())

        def line_updater(mob):
            z = z_tracker.get_value()
            points = get_line_points(z)
            if points is None:
                mob.set_stroke(opacity=0)
            else:
                mob.set_points_as_corners(points)
                mob.set_stroke(opacity=1)

        def scanner_updater(mob):
            z = z_tracker.get_value()
            points = get_line_points(z)
            if points is None:
                mob.set_opacity(0)
            else:
                # Place scanner at the center of the current diagonal line
                center = (points[0] + points[1]) / 2
                mob.move_to(center)
                mob.set_opacity(0.8)

        sum_line.add_updater(line_updater)
        scanner_icon.add_updater(scanner_updater)
        
        self.play(Create(sum_line), FadeIn(scanner_icon))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_ACTIVE)

        # Slide scanner line smoothly across grid
        self.play(z_tracker.animate.set_value(0), run_time=1.5)
        self.play(z_tracker.animate.set_value(10), run_time=4, rate_func=linear)
        self.play(z_tracker.animate.set_value(3), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_ACTIVE)

        # Highlight contributing points for Z=3: (0,3), (1,2), (2,1), (3,0)
        highlight_dots = VGroup(*[
            Dot(axes.c2p(i, 3-i, 0), color=YELLOW, radius=0.1)
            for i in range(4)
        ])
        
        self.play(LaggedStart(*[FadeIn(d, scale=2) for d in highlight_dots], lag_ratio=0.2))
        self.wait(2)

        # Finishing
        self.lecture[4].set_color(WHITE)
        self.play(FadeOut(highlight_dots), FadeOut(formula), FadeOut(sum_line), FadeOut(scanner_icon))
        self.wait(1)
