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
        # Title and Lecture Lines
        title = "The Power Rule: The Geometry of Growing Squares"
        lines = [
            "Imagine a square with side x and area x-squared.",
            "As x grows, thin strips appear on the edges.",
            "A tiny corner piece also forms as it expands.",
            "The two side strips each have an area of x.",
            "The Power Rule proves this growth rate is 2x."
        ]
        self.setup_layout(title, lines)

        # Colors
        BLUE_STRIP = "#ADD8E6"
        RED_CORNER = "#FF0000"
        GROWTH_COLOR = "#FFFF00"

        # Assets
        SQUARE_ASSET = "/mmfs1/data/home/jthen/Code2Video/assets/icon/square.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Create Main Square from Asset
        main_square = SVGMobject(SQUARE_ASSET).set_color(WHITE).set_width(2.0)
        self.place_in_area(main_square, "C2", "D3")
        
        # Area label
        area_label = MarkupText("x<sup>2</sup>", color=WHITE)
        self.place_in_area(area_label, "C2", "D3", scale_factor=0.8)
        
        # Side labels
        side_x_bottom = Text("x", color=WHITE)
        self.place_at_grid(side_x_bottom, "E2", scale_factor=0.7)
        side_x_bottom.shift(RIGHT * 0.5) 
        
        side_x_left = Text("x", color=WHITE)
        self.place_at_grid(side_x_left, "C1", scale_factor=0.7)
        side_x_left.shift(DOWN * 0.5) 
        
        self.play(Create(main_square), Write(area_label))
        self.play(Write(side_x_bottom), Write(side_x_left))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(BLUE_STRIP)
        )
        
        # Strips
        dx = 0.3
        top_strip = Rectangle(width=2.0, height=dx, color=BLUE_STRIP, fill_opacity=0.5, stroke_width=1)
        right_strip = Rectangle(width=dx, height=2.0, color=BLUE_STRIP, fill_opacity=0.5, stroke_width=1)
        
        top_strip.next_to(main_square, UP, buff=0)
        right_strip.next_to(main_square, RIGHT, buff=0)

        # dx labels with fixed positions (Issue 33 and 34)
        dx_label_top = Text("dx", color=BLUE_STRIP)
        self.place_at_grid(dx_label_top, "A2", scale_factor=0.6)
        dx_label_top.shift(RIGHT * 0.5)

        dx_label_right = Text("dx", color=BLUE_STRIP)
        self.place_at_grid(dx_label_right, "C5", scale_factor=0.6)
        dx_label_right.shift(DOWN * 0.5)

        self.play(
            FadeIn(top_strip, shift=UP*0.1),
            FadeIn(right_strip, shift=RIGHT*0.1)
        )
        self.play(Write(dx_label_top), Write(dx_label_right))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(RED_CORNER)
        )

        # Corner square from Asset (Issue 27)
        corner_square = SVGMobject(SQUARE_ASSET).set_color(RED_CORNER).set_width(dx)
        corner_square.next_to(right_strip, UP, buff=0)

        corner_label = MarkupText("dx<sup>2</sup>", color=RED_CORNER)
        self.place_at_grid(corner_label, "B4", scale_factor=0.4)

        self.play(Create(corner_square))
        self.play(Write(corner_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(GRAY),
            self.lecture[3].animate.set_color(BLUE_STRIP)
        )
        
        # Area labels for strips
        strip_label_1 = Text("x · dx", color=BLUE_STRIP)
        self.place_at_grid(strip_label_1, "B3", scale_factor=0.5)
        
        strip_label_2 = Text("x · dx", color=BLUE_STRIP)
        self.place_at_grid(strip_label_2, "D4", scale_factor=0.5)
        strip_label_2.rotate(-PI/2)

        self.play(Write(strip_label_1), Write(strip_label_2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(GRAY),
            self.lecture[4].animate.set_color(GROWTH_COLOR)
        )

        # Total growth and power rule summary (Issue 35)
        growth_text = Text("Total Growth ≈ 2x · dx", color=GROWTH_COLOR, font_size=24)
        self.place_in_area(growth_text, "E4", "E6", scale_factor=1.0)
        
        power_rule = MarkupText("d/dx(x<sup>2</sup>) = 2x", color=GROWTH_COLOR)
        self.place_in_area(power_rule, "F1", "F6", scale_factor=1.0)

        self.play(Write(growth_text))
        self.play(FadeIn(power_rule, shift=UP*0.2))
        
        self.play(
            strip_label_1.animate.set_color(GROWTH_COLOR),
            strip_label_2.animate.set_color(GROWTH_COLOR),
            rate_func=there_and_back
        )
        
        self.wait(2)
