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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initialize Scene
        self.setup_layout(
            "The Big Picture: Learning as Trial and Error",
            [
                "Neural networks learn through simple trial and error.",
                "The goal is to minimize prediction errors.",
                "Adjusting internal weights brings us closer to truth."
            ]
        )

        # Define Colors
        COLOR_ROBOT = "#FFD700"
        COLOR_TARGET = "#FF4500"
        COLOR_ERROR = "#FF0000"
        COLOR_SLIDER = "#00FF00"

        # Define Assets
        # Robot Archer [Asset: robot.svg and archer.svg]
        robot_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg", color=COLOR_ROBOT)
        archer_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/archer.svg", color=COLOR_ROBOT)
        robot_archer = VGroup(robot_svg, archer_svg).arrange(RIGHT, buff=0.2)
        # Fix Issue 47: Robot at C1
        self.place_at_grid(robot_archer, "C1", scale_factor=0.6)

        # Target Bullseye
        target = VGroup(
            Circle(radius=0.6, color=COLOR_TARGET, fill_opacity=0.2, stroke_width=2),
            Circle(radius=0.4, color=COLOR_TARGET, fill_opacity=0.4, stroke_width=2),
            Circle(radius=0.2, color=COLOR_TARGET, fill_opacity=1.0, stroke_width=0)
        )
        # Fix Issue 47: Target at C6, scale 0.9
        self.place_at_grid(target, "C6", scale_factor=0.9)

        # Arrow Asset
        # Initially hidden at robot's position
        arrow = Arrow(
            start=self.grid["C1"],
            end=self.grid["C1"] + RIGHT*0.1,
            color=WHITE,
            buff=0,
            stroke_width=4
        )

        # Error UI
        miss_pos = self.grid["A6"]
        target_pos = self.grid["C6"]
        error_line = DashedLine(miss_pos, target_pos, color=COLOR_ERROR)
        error_text = Text("Error", color=COLOR_ERROR, font_size=20)
        # Fix Issue 49: Error text at B6, scale 0.8
        self.place_at_grid(error_text, "B6", scale_factor=0.8)

        # Slider Asset
        slider_rail = Line(LEFT, RIGHT, color=WHITE).scale(0.8)
        slider_knob = Dot(color=COLOR_SLIDER)
        slider_label = Text("Arm Tension", color=COLOR_SLIDER, font_size=18).next_to(slider_rail, UP, buff=0.1)
        slider = VGroup(slider_rail, slider_knob, slider_label)
        # Fix Issue 48: Slider at F1
        self.place_at_grid(slider, "F1", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Neural networks learn through simple trial and error.
        self.play(self.lecture[0].animate.set_color(COLOR_ROBOT))
        self.play(FadeIn(robot_archer), FadeIn(target))
        
        # Shot 1: Misses
        self.add(arrow)
        self.play(arrow.animate.put_start_and_end_on(self.grid["C1"], miss_pos), run_time=1.2, rate_func=rush_into)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # The goal is to minimize prediction errors.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_ERROR)
        )
        
        # Highlight Error
        self.play(Create(error_line), Write(error_text))
        self.play(FadeIn(slider))
        # Move knob to show adjustment
        self.play(slider_knob.animate.shift(RIGHT * 0.4))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Adjusting internal weights brings us closer to truth.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_TARGET)
        )
        
        # Shot 2: Hits
        self.play(
            FadeOut(error_line),
            FadeOut(error_text),
            arrow.animate.put_start_and_end_on(self.grid["C1"], target_pos),
            run_time=1.2,
            rate_func=rush_into
        )
        self.play(Indicate(target, color=COLOR_TARGET))
        self.wait(2)

        # Final color reset
        self.play(self.lecture[2].animate.set_color(WHITE))
