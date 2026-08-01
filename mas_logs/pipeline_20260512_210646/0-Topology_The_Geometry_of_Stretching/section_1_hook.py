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

class Section1HookScene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'Topology is often called "rubber sheet geometry."',
            'We can deform shapes without tearing or gluing.',
            'This mug smoothly transforms into a donut.',
            'Both objects share exactly one fundamental hole.',
            'In topology, these two shapes are equivalent.'
        ]
        self.setup_layout("The Mug and the Donut Paradox", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Load Mug SVG [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/mug.svg]
        mug = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/mug.svg")
        mug.set_color("#D2B48C")
        # Fix from Issue 38: Place in area C3-F5
        self.place_in_area(mug, "C3", "F5", scale_factor=1.5)
        
        mug_label = Text("Mug", font_size=24, color=WHITE)
        # Fix from Issue 40: Place at grid B4
        self.place_at_grid(mug_label, "B4", scale_factor=0.8)

        self.play(FadeIn(mug), Write(mug_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Intermediate: Plate and handle
        # Represent the "rubber sheet" stretching process
        plate = Rectangle(height=0.3, width=2.2, fill_opacity=1, color="#D2B48C")
        handle_arc = Arc(radius=0.5, start_angle=-PI/2, angle=PI, stroke_width=12, color="#D2B48C")
        handle_arc.shift(RIGHT * 1.0)
        intermediate_shape = VGroup(plate, handle_arc)
        self.place_in_area(intermediate_shape, "D3", "F5", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(mug, intermediate_shape),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Load Donut SVG [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/donut.svg]
        donut = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/donut.svg")
        donut.set_color("#FF69B4")
        self.place_in_area(donut, "C3", "F5", scale_factor=1.5)
        
        donut_label = Text("Donut", font_size=24, color=WHITE)
        self.place_at_grid(donut_label, "B4", scale_factor=0.8)

        self.play(
            ReplacementTransform(intermediate_shape, donut),
            ReplacementTransform(mug_label, donut_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )

        # Flash the hole area
        hole_highlight = Circle(radius=0.4, color="#FFFF00", stroke_width=10)
        hole_highlight.move_to(donut.get_center())
        
        self.play(Flash(hole_highlight, color="#FFFF00", flash_radius=0.7))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )

        equiv_text = Text("Topologically Equivalent", font_size=24, color="#00FF00")
        # Fix from Issue 39: Place in area F3-F5
        self.place_in_area(equiv_text, "F3", "F5", scale_factor=0.8)
        
        self.play(Write(equiv_text))
        self.wait(2)

        # Reset colors
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
