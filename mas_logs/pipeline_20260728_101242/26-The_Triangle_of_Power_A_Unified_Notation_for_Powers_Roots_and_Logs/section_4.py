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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("One Triangle, Three Questions", [
            "Each vertex represents a different mathematical question.",
            "Missing the bottom-right? That is a power calculation.",
            "Missing the bottom-left? That is a root calculation.",
            "Missing the top? That is a logarithm calculation.",
            "The triangle stays fixed; only the unknown changes."
        ])

        # Define triangle vertices based on grid
        # Issue 39: Centering apex at the middle of B3 and B4
        v_top = MathTex("3", color=RED)
        self.place_in_area(v_top, "B3", "B4", scale_factor=0.7)
        top_pos = v_top.get_center()

        # Issue 38 & 40: Move v_br to E5 and scale vertex labels to 0.7
        v_bl = MathTex("2", color=GREEN)
        self.place_at_grid(v_bl, "E2", scale_factor=0.7)
        bl_pos = v_bl.get_center()

        v_br = MathTex("8", color=BLUE)
        self.place_at_grid(v_br, "E5", scale_factor=0.7)
        br_pos = v_br.get_center()

        # Triangle connecting the labels
        triangle = Polygon(top_pos, bl_pos, br_pos, color=WHITE, stroke_width=4)

        # Robotic arm components
        # Issue 40: Scale arm_hand to 0.7
        arm_joint = Dot(self.grid["A6"] + RIGHT * 2, color="#C0C0C0") # Hidden joint anchor
        arm_hand = Square(side_length=1.0, fill_opacity=1, fill_color="#C0C0C0", stroke_color=WHITE)
        arm_hand.scale(0.7)
        arm_hand.move_to(self.grid["A6"] + UP * 2) # Start off-screen
        
        # Persistent arm line using updater
        arm_line = Line(arm_joint.get_center(), arm_hand.get_center(), color="#C0C0C0", stroke_width=6)
        arm_line.add_updater(lambda l: l.put_start_and_end_on(arm_joint.get_center(), arm_hand.get_center()))
        
        # Question mark for the arm to "carry"
        q_mark = MathTex("?", font_size=40, color=BLACK)
        q_mark.add_updater(lambda m: m.move_to(arm_hand.get_center()))

        robotic_arm = VGroup(arm_joint, arm_line, arm_hand)

        # === Animation for Lecture Line 1 ===
        # Each vertex represents a different mathematical question.
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Create(triangle), run_time=1)
        self.play(Write(v_top), Write(v_bl), Write(v_br))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Missing the bottom-right? That is a power calculation.
        self.play(self.lecture[1].animate.set_color(BLUE))
        q_mark.set_color(BLUE)
        self.add(robotic_arm, q_mark)
        # Move hand to cover br_pos
        self.play(arm_hand.animate.move_to(br_pos), run_time=1.5)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Missing the bottom-left? That is a root calculation.
        self.play(self.lecture[2].animate.set_color(GREEN))
        # Move hand to cover bl_pos
        self.play(
            arm_hand.animate.move_to(bl_pos),
            q_mark.animate.set_color(GREEN),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Missing the top? That is a logarithm calculation.
        self.play(self.lecture[3].animate.set_color(RED))
        # Move hand to cover top_pos
        self.play(
            arm_hand.animate.move_to(top_pos),
            q_mark.animate.set_color(RED),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # The triangle stays fixed; only the unknown changes.
        self.play(self.lecture[4].animate.set_color(WHITE))
        # Arm retracts off screen
        self.play(
            arm_hand.animate.move_to(self.grid["A6"] + RIGHT * 3 + UP * 2),
            FadeOut(q_mark),
            run_time=1.5
        )
        self.remove(robotic_arm)
        # Triangle glows white (briefly emphasize it stays fixed)
        self.play(triangle.animate.set_stroke(width=8), run_time=0.5)
        self.play(triangle.animate.set_stroke(width=4), run_time=0.5)
        self.wait(2)

# Set issues as resolved
# update_issue(38, under_review=True, resolution_note="Moved v_br to E5 and updated vertex/hand scale to 0.7.")
# update_issue(39, under_review=True, resolution_note="Apex v_top centered between B3 and B4 using place_in_area.")
# update_issue(40, under_review=True, resolution_note="Scaled labels and unknown box to 0.7 as requested.")

import numpy as np
from manim import *

# Final batch update of issues
# replace_code(...)
# update_issue(38, under_review=True, resolution_note="Applied grid position fix for v_br and scale adjustment.")
# update_issue(39, under_review=True, resolution_note="Applied centering fix for v_top using place_in_area.")
# update_issue(40, under_review=True, resolution_note="Scaled vertex labels and arm_hand to 0.7.")
