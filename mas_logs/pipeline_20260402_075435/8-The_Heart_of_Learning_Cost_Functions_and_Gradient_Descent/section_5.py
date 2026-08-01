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
        # Initial Setup
        title = "The Learning Rate: How Big is the Step?"
        lines = [
            "The Learning Rate decides the size of each step.",
            "Too high, and Pixel leaps over the goal.",
            "Too low, and progress becomes painfully slow."
        ]
        self.setup_layout(title, lines)

        # Colors for mapping
        COLOR_LR = YELLOW
        COLOR_HIGH = RED
        COLOR_LOW = BLUE

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_LR))

        # Setup High Alpha Viz (Top Half)
        axes_high = Axes(
            x_range=[-3, 3, 1], 
            y_range=[0, 9, 1], 
            x_length=4.0, 
            y_length=2.5, 
            axis_config={"include_tip": False, "color": GREY_D}
        )
        curve_high = axes_high.plot(lambda x: x**2, color=COLOR_HIGH, x_range=[-2.8, 2.8])
        label_high = Text("High Learning Rate (Alpha)", font_size=16, color=COLOR_HIGH)
        high_group = VGroup(axes_high, curve_high, label_high).arrange(UP, buff=0.1)
        
        # Fix for Issue #39 and #40: Shifted down and scaled
        self.place_in_area(high_group, "B1", "C6", scale_factor=0.8)

        # Setup Low Alpha Viz (Bottom Half)
        axes_low = Axes(
            x_range=[-3, 3, 1], 
            y_range=[0, 9, 1], 
            x_length=4.0, 
            y_length=2.5, 
            axis_config={"include_tip": False, "color": GREY_D}
        )
        curve_low = axes_low.plot(lambda x: x**2, color=COLOR_LOW, x_range=[-2.8, 2.8])
        label_low = Text("Low Learning Rate (Alpha)", font_size=16, color=COLOR_LOW)
        low_group = VGroup(axes_low, curve_low, label_low).arrange(UP, buff=0.1)
        
        # Fix for Issue #41: Shifted down to bottom area and scaled
        self.place_in_area(low_group, "E1", "F6", scale_factor=0.8)

        self.play(Create(high_group), Create(low_group), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HIGH)
        )

        # "Pixel" for high alpha
        pixel_high = Dot(color=WHITE, radius=0.06)
        pixel_high.move_to(axes_high.c2p(2.2, 4.84))
        self.add(pixel_high)

        # Leap sequence: overshoot valley
        leap_points = [
            axes_high.c2p(-2.0, 4.0),
            axes_high.c2p(1.8, 3.24),
            axes_high.c2p(-1.4, 1.96),
            axes_high.c2p(0.8, 0.64)
        ]

        for pt in leap_points:
            self.play(pixel_high.animate.move_to(pt), run_time=0.6, rate_func=slow_into)
        
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_LOW)
        )

        # "Pixel" for low alpha
        pixel_low = Dot(color=WHITE, radius=0.06)
        pixel_low.move_to(axes_low.c2p(2.2, 4.84))
        self.add(pixel_low)

        # Shuffle sequence: tiny steps
        shuffle_points = [
            axes_low.c2p(2.0, 4.0),
            axes_low.c2p(1.8, 3.24),
            axes_low.c2p(1.6, 2.56),
            axes_low.c2p(1.4, 1.96),
            axes_low.c2p(1.2, 1.44)
        ]

        for pt in shuffle_points:
            self.play(pixel_low.animate.move_to(pt), run_time=0.4, rate_func=linear)

        self.wait(2)
