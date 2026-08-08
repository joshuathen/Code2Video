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
        # Define content based on storyboard
        title = "Vector Addition: The Tip-to-Tail Method"
        lecture_lines = [
            "Adding vectors combines two movements into one.",
            "Place the second vector's tail at the first's tip.",
            "The new vector connects the start to the end.",
            "This result represents the sum of both forces.",
            "Algebraically, we just add the corresponding components."
        ]
        self.setup_layout(title, lecture_lines)

        # === Setup for Vectors ===
        # Initial positions on grid (will be adjusted by vector_group layout)
        vec_a = Arrow(self.grid["D2"], self.grid["B3"], color="#00FF00", buff=0)
        label_a = MathTex(r"\vec{A}", color="#00FF00").scale(0.8)
        label_a.next_to(vec_a, LEFT, buff=0.1)
        
        vec_b = Arrow(self.grid["E4"], self.grid["D6"], color="#FF0000", buff=0)
        label_b = MathTex(r"\vec{B}", color="#FF0000").scale(0.8)
        label_b.next_to(vec_b, RIGHT, buff=0.1)

        # Issue 23 Fix: Scale and place the initial vector setup to utilize full drawing space
        vector_group = VGroup(vec_a, label_a, vec_b, label_b)
        self.place_in_area(vector_group, 'A2', 'D5', scale_factor=1.1)
        
        # Capture updated coordinates after layout adjustment
        start_a = vec_a.get_start()
        end_a = vec_a.get_end()
        disp_b = vec_b.get_vector()

        # === Animation for Lecture Line 1 ===
        # "Adding vectors combines two movements into one."
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.play(Create(vec_a), Write(label_a))
        self.play(Create(vec_b), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Place the second vector's tail at the first's tip."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FF0000")
        )
        
        # Move vector B tail to tip of vector A
        # center = end_a + (vector_displacement / 2)
        target_center_b = end_a + (disp_b / 2)
        self.play(
            vec_b.animate.move_to(target_center_b),
            label_b.animate.next_to(end_a + disp_b, RIGHT, buff=0.1)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The new vector connects the start to the end."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        # Draw dashed line from start of A to final tip of B
        dashed_res = DashedLine(start_a, end_a + disp_b, color="#FFFF00")
        self.play(Create(dashed_res))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "This result represents the sum of both forces."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#FFFF00")
        )
        
        resultant_vec = Arrow(start_a, end_a + disp_b, color="#FFFF00", buff=0)
        label_res = MathTex(r"\vec{A} + \vec{B}", color="#FFFFFF").scale(0.8)
        # Offset label to avoid overlap with the vector path
        label_res.next_to(resultant_vec.get_center(), DR, buff=0.2)
        
        self.play(
            ReplacementTransform(dashed_res, resultant_vec),
            Write(label_res)
        )
        # Emphasis scale
        self.play(label_res.animate.scale(1.3))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Algebraically, we just add the corresponding components."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#87CEEB")
        )
        
        math_box = MathTex(
            r"\begin{bmatrix} x_A \\ y_A \end{bmatrix} + \begin{bmatrix} x_B \\ y_B \end{bmatrix} = \begin{bmatrix} x_A + x_B \\ y_A + y_B \end{bmatrix}",
            color="#87CEEB"
        )
        # Issue 21 & 22 combined fix: Better centering and avoiding overlap with labels/vectors
        self.place_in_area(math_box, 'E1', 'F6', scale_factor=0.7)
        
        self.play(FadeIn(math_box, shift=UP))
        self.wait(3)
