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
        # Data from storyboard
        title_text = "The Fragmented Language of Math"
        lecture_lines = [
            "Mathematics uses different symbols for the same relationship.",
            "Exponents, radicals, and logs often look completely unrelated.",
            "This fragmentation makes simple concepts feel unnecessarily confusing."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        BLUE_EX = "#00CCFF"
        ORANGE_RA = "#FF9900"
        PURPLE_LO = "#CC00FF"
        GRAY_MA = "#808080"
        WHITE_PIX = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(BLUE_EX))
        
        # Issue 25: Show 2^3 = 8 at A1-A2 to avoid machine obstruction
        exp_eq = MathTex("2^3 = 8", color=BLUE_EX)
        self.place_in_area(exp_eq, "A1", "A2", scale_factor=0.8)
        
        self.play(Write(exp_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(ORANGE_RA)
        )
        
        # Issue 26 & 27: Position rad_eq and log_eq consistently on Row A
        rad_eq = MathTex(r"\sqrt[3]{8} = 2", color=ORANGE_RA)
        self.place_in_area(rad_eq, "A3", "A4", scale_factor=0.8)
        
        log_eq = MathTex(r"\log_2 8 = 3", color=PURPLE_LO)
        self.place_in_area(log_eq, "A5", "A6", scale_factor=0.8)
        
        self.play(
            Write(rad_eq),
            Write(log_eq)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GRAY_MA)
        )
        
        # Cluttered Machine (B1-E6)
        machine_body = RoundedRectangle(corner_radius=0.2, width=5.0, height=3.5, color=GRAY_MA, fill_opacity=0.1)
        gear1 = Star(n=8, outer_radius=0.4, inner_radius=0.2, color=GRAY_MA, fill_opacity=0.3)
        gear2 = Star(n=10, outer_radius=0.5, inner_radius=0.3, color=GRAY_MA, fill_opacity=0.3)
        pipe1 = Line(LEFT*2.5, RIGHT*2.5, color=GRAY_MA, stroke_width=6)
        lever = Line(ORIGIN, UP*0.8, color=GRAY_MA, stroke_width=8)
        knob = Circle(radius=0.15, color=RED, fill_opacity=1)
        
        machine = VGroup(machine_body, gear1, gear2, pipe1, lever, knob)
        # Position sub-components relative to machine center (will be moved by place_in_area)
        gear1.move_to(machine_body.get_center() + UP*0.8 + LEFT*1.5)
        gear2.move_to(machine_body.get_center() + DOWN*0.6 + RIGHT*1.2)
        pipe1.move_to(machine_body.get_center() + UP*1.4)
        lever.move_to(machine_body.get_center() + RIGHT*2.2 + DOWN*0.5)
        knob.move_to(lever.get_end())
        
        machine.set_z_index(-1)
        self.place_in_area(machine, "B1", "E6", scale_factor=1.0)
        
        # Pixel the Penguin
        pixel_body = Ellipse(width=0.6, height=0.8, color=WHITE_PIX, fill_opacity=1).set_stroke(GRAY, 1)
        pixel_belly = Ellipse(width=0.4, height=0.5, color=WHITE, fill_opacity=1).move_to(pixel_body.get_center() + DOWN*0.1)
        eye_l = Dot(radius=0.04, color=BLACK).move_to(pixel_body.get_center() + UP*0.2 + LEFT*0.12)
        eye_r = Dot(radius=0.04, color=BLACK).move_to(pixel_body.get_center() + UP*0.2 + RIGHT*0.12)
        beak = Triangle(color=ORANGE, fill_opacity=1).scale(0.08).rotate(PI).move_to(pixel_body.get_center() + UP*0.05)
        pixel = VGroup(pixel_body, pixel_belly, eye_l, eye_r, beak)
        
        # Confused question mark
        q_mark = Text("?", color=WHITE_PIX).scale(1.2).next_to(pixel, UP, buff=0.2)
        pixel_confused = VGroup(pixel, q_mark)
        self.place_at_grid(pixel_confused, "F1", scale_factor=0.8)
        
        self.play(FadeIn(machine), run_time=2)
        self.play(FadeIn(pixel_confused))
        
        # Final animation: rotate gears and make penguin bounce
        self.play(
            Rotate(gear1, angle=2*PI, run_time=4, rate_func=linear),
            Rotate(gear2, angle=-2*PI, run_time=4, rate_func=linear),
            pixel_confused.animate(run_time=1, rate_func=there_and_back).shift(UP*0.2),
        )
        self.wait(2)
