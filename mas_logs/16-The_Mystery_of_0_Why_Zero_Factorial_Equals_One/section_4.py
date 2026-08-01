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
        title = "The Combinatorial Logic: Arranging 'Nothing'"
        lines = [
            "Combinatorics defines factorials as ways to arrange objects.",
            "Robot B-0 is asked to arrange zero blocks.",
            "He does nothing, leaving the space completely empty.",
            "There is exactly one way to do absolutely nothing.",
            "Thus, the null arrangement counts as one."
        ]
        self.setup_layout(title, lines)

        # Custom stylized Robot B-0 since no path was provided for the asset
        robot_head = Square(side_length=0.6, fill_opacity=1, color=GRAY_B)
        robot_eye_l = Dot(radius=0.05, color=BLACK).move_to(robot_head.get_center() + [-0.15, 0.1, 0])
        robot_eye_r = Dot(radius=0.05, color=BLACK).move_to(robot_head.get_center() + [0.15, 0.1, 0])
        robot_mouth = Line([-0.1, -0.1, 0], [0.1, -0.1, 0], color=BLACK)
        robot_body = RoundedRectangle(corner_radius=0.1, height=1.0, width=0.8, fill_opacity=1, color=GRAY_A)
        robot_label = Text("B-0", font_size=16, color=BLACK).move_to(robot_body.get_center())
        robot_head.next_to(robot_body, UP, buff=0.05)
        robot_eye_l.move_to(robot_head.get_center() + [-0.15, 0.1, 0])
        robot_eye_r.move_to(robot_head.get_center() + [0.15, 0.1, 0])
        robot_mouth.move_to(robot_head.get_center() + [0, -0.15, 0])
        robot = VGroup(robot_body, robot_head, robot_eye_l, robot_eye_r, robot_mouth, robot_label)

        # Table Surface
        table = Rectangle(width=3.5, height=0.2, fill_opacity=1, color=DARK_BROWN)
        table_leg1 = Rectangle(width=0.1, height=1.0, fill_opacity=1, color=DARK_BROWN).next_to(table, DOWN, buff=0, aligned_edge=LEFT).shift(RIGHT*0.5)
        table_leg2 = Rectangle(width=0.1, height=1.0, fill_opacity=1, color=DARK_BROWN).next_to(table, DOWN, buff=0, aligned_edge=RIGHT).shift(LEFT*0.5)
        table_surface = VGroup(table, table_leg1, table_leg2)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_in_area(robot, "B1", "D2", scale_factor=0.8)
        self.place_in_area(table_surface, "E3", "F6", scale_factor=0.8)
        self.play(FadeIn(robot), FadeIn(table_surface))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        task_text = Text("Task: Arrange 0 blocks", font_size=20, color=WHITE)
        self.place_at_grid(task_text, "A3", scale_factor=1.0)
        
        self.play(Write(task_text))
        self.play(robot_head.animate.rotate(-0.2, about_point=robot_head.get_center())) # Robot looks at table
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        # Highlight empty space
        glow_circle = Circle(radius=0.6, color=YELLOW).set_stroke(width=8).set_opacity(0.5)
        self.place_at_grid(glow_circle, "D4", scale_factor=1.0)
        
        self.play(Create(glow_circle))
        self.play(Indicate(glow_circle, color=YELLOW, scale_factor=1.2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        empty_set_text = Text("The Empty Set", font_size=24, color="#00FFFF")
        self.place_at_grid(empty_set_text, "C4", scale_factor=1.0)
        
        self.play(FadeIn(empty_set_text, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        one_way_text = Text("One way to do nothing", font_size=24, color=WHITE)
        factorial_result = Text("0! = 1", font_size=48, color=WHITE)
        self.place_in_area(one_way_text, "B3", "B5", scale_factor=1.0)
        self.place_in_area(factorial_result, "B3", "B5", scale_factor=1.0)
        
        self.play(FadeOut(task_text))
        self.play(Write(one_way_text))
        self.wait(1)
        self.play(ReplacementTransform(one_way_text, factorial_result))
        self.play(Indicate(factorial_result))
        
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
