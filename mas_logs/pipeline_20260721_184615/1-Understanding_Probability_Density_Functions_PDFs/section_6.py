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

class Section6Scene(TeachingScene):
    def construct(self):
        # Section 6: Application: The Uniform Distribution
        title_text = "Application: The Uniform Distribution"
        lecture_lines = [
            "Imagine a bus arriving anytime within ten minutes.",
            "This creates a flat, rectangular probability shape.",
            "Calculate simple areas to find the wait time."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        YELLOW_COLOR = "#FAFAD2"
        BLUE_COLOR = "#00BFFF"
        AXIS_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW_COLOR))

        # Setup Axes relative to grid
        # Horizontal: 0 to 10 (Grid cols 1 to 6)
        # Vertical: 0 to 0.15 (Grid rows E to B)
        origin_pos = self.grid["E1"]
        x_axis_end = self.grid["E6"]
        y_axis_end = self.grid["B1"]

        axes = Axes(
            x_range=[0, 10.5, 2],
            y_range=[0, 0.15, 0.05],
            x_length=abs(x_axis_end[0] - origin_pos[0]),
            y_length=abs(y_axis_end[1] - origin_pos[1]),
            axis_config={"color": AXIS_COLOR, "include_tip": True},
            tips=True
        ).move_to(origin_pos, aligned_edge=DOWN+LEFT)
        
        x_label = Text("Minutes", font_size=20).next_to(axes.x_axis, DOWN, buff=0.2)
        y_label = Text("P(x)", font_size=20).next_to(axes.y_axis, LEFT, buff=0.2)

        # Draw a flat rectangle from 0 to 10 with height 0.1
        rect_width = axes.x_axis.get_unit_size() * 10
        rect_height = axes.y_axis.get_unit_size() * 0.1
        
        full_rect = Rectangle(
            width=rect_width,
            height=rect_height,
            stroke_color=YELLOW_COLOR,
            fill_color=YELLOW_COLOR,
            fill_opacity=0.3
        ).move_to(axes.c2p(5, 0.05))

        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bus.svg]
        bus_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/bus.svg"
        bus = SVGMobject(bus_asset_path).scale(0.3)
        # Place bus at the beginning of the path (x=0)
        bus.move_to(axes.c2p(0, 0), aligned_edge=DOWN)

        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(full_rect), DrawBorderThenFill(bus))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW_COLOR)
        )

        # Shade from 0 to 5
        shade_rect = Rectangle(
            width=axes.x_axis.get_unit_size() * 5,
            height=rect_height,
            stroke_width=0,
            fill_color=BLUE_COLOR,
            fill_opacity=0.6
        ).move_to(axes.c2p(2.5, 0.05))

        # Label base '5' and height '0.1' with arrows (#FAFAD2)
        # Base arrow
        base_arrow = DoubleArrow(
            start=axes.c2p(0, -0.05),
            end=axes.c2p(5, -0.05),
            buff=0,
            color=YELLOW_COLOR,
            stroke_width=2,
            tip_length=0.1
        )
        base_label = Text("5", font_size=20, color=YELLOW_COLOR).next_to(base_arrow, DOWN, buff=0.1)

        # Height arrow
        height_arrow = DoubleArrow(
            start=axes.c2p(-0.7, 0),
            end=axes.c2p(-0.7, 0.1),
            buff=0,
            color=YELLOW_COLOR,
            stroke_width=2,
            tip_length=0.1
        )
        
        # Issue 35: scale_factor=0.6 at C1
        height_label = Text("0.1", font_size=20, color=YELLOW_COLOR)
        self.place_at_grid(height_label, "C1", scale_factor=0.6)
        # Shift slightly to be left of the arrow/axis
        height_label.shift(LEFT * 0.6)

        self.play(
            FadeIn(shade_rect),
            Create(base_arrow),
            Write(base_label),
            Create(height_arrow),
            Write(height_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(BLUE_COLOR)
        )

        # Issue 36: Display calculation '5 * 0.1 = 0.5' in area A4-B6, scale_factor=0.8
        calc_text = MathTex("5 \\times 0.1 =", "0.5", font_size=32, color=WHITE)
        self.place_in_area(calc_text, "A4", "B6", scale_factor=0.8)

        self.play(Write(calc_text))
        # Highlight result in bright blue (#00BFFF)
        self.play(Indicate(calc_text[1], color=BLUE_COLOR))
        self.play(calc_text[1].animate.set_color(BLUE_COLOR))
        
        self.wait(3)
