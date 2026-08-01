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

class Section2Scene(TeachingScene):
    def construct(self):
        # Section Title and Lecture Lines
        title_text = "The B.I.N.S. Criteria"
        lecture_lines = [
            "Binomial distributions must meet four specific criteria, called BINS.",
            "\"B\" stands for Binary: outcomes are either success or failure.",
            "\"I\" means Independent: one trial doesn't affect the next.",
            "\"N\" is for Number: there's a fixed amount of trials.",
            "\"S\" stands for Same: the probability p remains constant."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_ACRONYM = "#FFFFFF"
        COLOR_CRITERIA = "#AAAAAA"
        COLOR_ROBOT = "#888888"
        COLOR_SUCCESS = "#55FF55"
        COLOR_HIGHLIGHT = "#FFFF00"
        COLOR_VALUES = "#55AAFF"
        
        # --- Mobject Creation ---
        # Acronym B-I-N-S
        b_let = Text("B", color=COLOR_ACRONYM)
        i_let = Text("I", color=COLOR_ACRONYM)
        n_let = Text("N", color=COLOR_ACRONYM)
        s_let = Text("S", color=COLOR_ACRONYM)
        
        self.place_at_grid(b_let, "B2", scale_factor=0.8)
        self.place_at_grid(i_let, "C2", scale_factor=0.8)
        self.place_at_grid(n_let, "D2", scale_factor=0.8)
        self.place_at_grid(s_let, "E2", scale_factor=0.8)
        
        bins_vgroup = VGroup(b_let, i_let, n_let, s_let)
        
        # Criteria labels
        binary_label = Text("Binary", color=COLOR_CRITERIA)
        independent_label = Text("Independent", color=COLOR_CRITERIA)
        number_label = Text("Number", color=COLOR_CRITERIA)
        same_label = Text("Same", color=COLOR_CRITERIA)
        
        # Position labels at Col 3
        self.place_at_grid(binary_label, "B3", scale_factor=0.7)
        binary_label.align_to(self.grid["B3"], LEFT)
        
        self.place_at_grid(independent_label, "C3", scale_factor=0.7)
        independent_label.align_to(self.grid["C3"], LEFT)
        
        self.place_at_grid(number_label, "D3", scale_factor=0.7)
        number_label.align_to(self.grid["D3"], LEFT)
        
        self.place_at_grid(same_label, "E3", scale_factor=0.7)
        same_label.align_to(self.grid["E3"], LEFT)
            
        # Robot example
        # Resolving Issue 28: Robot at Col 5 to avoid overlap with long labels
        robot_body = Rectangle(height=1.0, width=0.7, color=COLOR_ROBOT, fill_opacity=0.6)
        robot_head = Square(side_length=0.35, color=COLOR_ROBOT, fill_opacity=0.6).next_to(robot_body, UP, buff=0.1)
        robot_arm = Line(ORIGIN, RIGHT * 0.4, color=COLOR_ROBOT).next_to(robot_body, RIGHT, buff=0, aligned_edge=UP)
        robot = VGroup(robot_body, robot_head, robot_arm)
        self.place_in_area(robot, "B5", "E5", scale_factor=0.8)
        
        # Ball and Checkmarks
        ball = Circle(radius=0.1, color=ORANGE, fill_opacity=1.0)
        checkmarks = VGroup(*[
            Text("✓", color=COLOR_SUCCESS, font_size=24) for _ in range(5)
        ]).arrange(RIGHT, buff=0.2)
        # Position checkmarks in Col 4, Row B
        self.place_at_grid(checkmarks, "B4", scale_factor=0.7)

        # Numerical Labels
        # Resolving Issue 29: Place n_label at D4
        n_label = Text("n = 5", color=COLOR_VALUES, font_size=22)
        self.place_at_grid(n_label, "D4", scale_factor=0.9)
        
        # Resolving Issue 30: Place p_label at E4
        p_label = Text("p = 0.70", color=COLOR_VALUES, font_size=22)
        self.place_at_grid(p_label, "E4", scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        # Binomial distributions must meet four specific criteria, called BINS.
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))
        self.play(Write(bins_vgroup))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "B" stands for Binary: outcomes are either success or failure.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HIGHLIGHT)
        )
        self.play(
            Indicate(b_let),
            FadeIn(binary_label, shift=RIGHT)
        )
        self.play(FadeIn(robot))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "I" means Independent: one trial doesn't affect the next.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )
        self.play(
            Indicate(i_let),
            FadeIn(independent_label, shift=RIGHT)
        )
        # Small visual activity for robot
        self.play(robot_arm.animate.rotate(0.2, about_point=robot_arm.get_start()), run_time=0.3)
        self.play(robot_arm.animate.rotate(-0.2, about_point=robot_arm.get_start()), run_time=0.3)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "N" is for Number: there's a fixed amount of trials.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_HIGHLIGHT)
        )
        self.play(
            Indicate(n_let),
            FadeIn(number_label, shift=RIGHT)
        )
        self.play(Write(n_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "S" stands for Same: the probability p remains constant.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_HIGHLIGHT)
        )
        self.play(
            Indicate(s_let),
            FadeIn(same_label, shift=RIGHT)
        )
        self.play(Write(p_label))
        
        # Shooting sequence
        ball.move_to(robot_arm.get_end())
        self.play(FadeIn(ball))
        target_point = self.grid["C6"]
        
        for i in range(5):
            self.play(
                ball.animate.move_to(target_point).set_opacity(0),
                run_time=0.4,
                rate_func=bezier([0, 0, 1, 1])
            )
            self.play(FadeIn(checkmarks[i], scale=1.2), run_time=0.2)
            ball.move_to(robot_arm.get_end()).set_opacity(1)
        
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
