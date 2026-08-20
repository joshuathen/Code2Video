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
        lecture_lines = [
            "The derivative is a transformation operator.",
            "It takes a function as input.",
            "It outputs the function's sensitivity.",
            "Sensitivity measures how output shifts with input.",
            "This describes the rate of change."
        ]
        self.setup_layout("The 'Operator' View: Derivatives as Transformation", lecture_lines)
        
        # Visual elements
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg]
        box = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg")
        self.place_at_grid(box, 'C4', scale_factor=0.9)
        label_box = Text("d/dx", font_size=24).move_to(box.get_center())
        
        input_f = MathTex("f(x)", color=YELLOW)
        self.place_at_grid(input_f, 'C3', scale_factor=1.0)
        
        output_f = MathTex("f'(x)", color=GREEN)
        self.place_at_grid(output_f, 'C5', scale_factor=1.0)
        
        arrow_in = Arrow(start=input_f.get_right(), end=box.get_left(), color=WHITE)
        arrow_out = Arrow(start=box.get_right(), end=output_f.get_left(), color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(box), Write(label_box))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(input_f), GrowArrow(arrow_in))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.play(GrowArrow(arrow_out), FadeIn(output_f))
        self.lecture[2].set_color(GREEN)

        # === Animation for Lecture Line 4 ===
        rect_sensitivity = SurroundingRectangle(output_f, color=ORANGE)
        self.play(Create(rect_sensitivity))
        self.lecture[3].set_color(ORANGE)

        # === Animation for Lecture Line 5 ===
        # Highlight using #FF4500 within the box
        box_highlight = box.copy().set_color("#FF4500")
        self.play(Transform(box, box_highlight), Indicate(output_f))
        self.lecture[4].set_color("#FF4500")
        
        self.wait(2)
