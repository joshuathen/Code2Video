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

class Section5Scene(TeachingScene):
    def construct(self):
        # 1. Setup layout with script and title
        title_text = "The Half-Circle Journey (π)"
        lecture_lines = [
            "If we set x to pi, we travel halfway.",
            "This half-circle journey starts at positive one.",
            "After distance pi, we land exactly at negative one."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Define custom colors
        GREEN_HIGHLIGHT = "#00FF00"
        RED_LANDING = "#FF0000"
        
        # Calculate the center of the visual area (grid A1-F6 center)
        tl_pos = self.grid["A1"]
        br_pos = self.grid["F6"]
        center_point = (tl_pos + br_pos) / 2
        
        # Setup Axes and Circle
        # We use a scale where 1 unit = 1.5 grid units
        axes = Axes(
            x_range=[-1.5, 1.5, 1], 
            y_range=[-1.5, 1.5, 1], 
            x_length=4.5, 
            y_length=4.5,
            axis_config={"color": BLUE_D, "stroke_width": 2},
            tips=True
        )
        # Align axes to the visualization area
        self.place_in_area(axes, 'A1', 'F6')
        
        unit_circle = Circle(radius=1.5, color=GRAY_D, stroke_width=2)
        unit_circle.move_to(center_point)

        # === Animation for Lecture Line 1 ===
        # Line: 'If we set x to pi, we travel halfway.'
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(
            Create(axes), 
            Create(unit_circle), 
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: 'This half-circle journey starts at positive one.'
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Define start point at (1,0) in our coordinate system
        start_dot = Dot(center_point + RIGHT * 1.5, color=WHITE)
        
        # Label "1" placed at grid B5 (moved from C5 to avoid axis overlap)
        label_one = Text("1", font_size=30)
        self.place_at_grid(label_one, "B5", scale_factor=0.8)
        
        self.play(FadeIn(start_dot), Write(label_one))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: 'After distance pi, we land exactly at negative one.'
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # The semi-circle path highlight (bright green)
        arc_highlight = Arc(
            radius=1.5, 
            start_angle=0, 
            angle=PI, 
            color=GREEN_HIGHLIGHT,
            arc_center=center_point,
            stroke_width=6
        )
        
        # The explorer dot representing our position
        explorer_dot = Dot(center_point + RIGHT * 1.5, color=WHITE, radius=0.12)
        
        # Red "-1" text at landing spot (Grid B2, scaled to match label_one)
        label_neg_one = Text("-1", font_size=30, color=RED_LANDING)
        self.place_at_grid(label_neg_one, "B2", scale_factor=0.8)
        
        # Transition from static start dot to moving explorer
        self.remove(start_dot)
        self.add(explorer_dot)
        
        # Animate the journey along the circle
        self.play(
            Create(arc_highlight),
            MoveAlongPath(explorer_dot, arc_highlight),
            run_time=4,
            rate_func=slow_into
        )
        
        # Reveal the result of the journey
        self.play(Write(label_neg_one))
        self.play(Indicate(label_neg_one, color=RED_LANDING))
        self.wait(2)
