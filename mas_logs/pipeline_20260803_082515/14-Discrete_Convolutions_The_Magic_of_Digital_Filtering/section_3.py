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
        self.setup_layout("The Tool: What is a Kernel?", [
            "- A kernel is a small array of weighting values.",
            "- It determines a neighbor's influence on a center point.",
            "- An averaging kernel smooths out noisy, jittery data."
        ])
        
        # Colors
        KERNEL_COLOR = "#98FB98"  # Light Green
        LABEL_COLOR = "#FFFFFF"
        INFLUENCE_COLOR = "#87CEEB"  # Sky Blue
        SMOOTH_COLOR = "#FFD700"  # Gold
        
        # Kernel Mobjects
        boxes = VGroup(*[Square(side_length=0.8, stroke_color=KERNEL_COLOR) for _ in range(3)]).arrange(RIGHT, buff=0.1)
        values = VGroup(*[MathTex(r"1/3", color=KERNEL_COLOR, font_size=24) for _ in range(3)])
        for i in range(3):
            values[i].move_to(boxes[i].get_center())
        
        kernel_group = VGroup(boxes, values)
        # Fix for Issue 30: Move kernel_group to row C and reduce scale
        self.place_in_area(kernel_group, 'C2', 'C4', scale_factor=1.0)
        
        kernel_label = Text("Averaging Kernel", font_size=20, color=LABEL_COLOR)
        # Fix for Issue 29: Move kernel_label to row B area for better vertical balance
        self.place_in_area(kernel_label, 'B2', 'B4')

        # === Animation for Lecture Line 1 ===
        # "A kernel is a small array of weighting values."
        self.play(self.lecture[0].animate.set_color(KERNEL_COLOR))
        self.play(
            Create(boxes),
            FadeIn(values),
            Write(kernel_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "It determines a neighbor's influence on a center point."
        self.play(self.lecture[1].animate.set_color(INFLUENCE_COLOR))
        
        # Pulse neighbors, then center to show "influence" flow
        self.play(
            boxes[0].animate.set_color(INFLUENCE_COLOR).scale(1.1),
            boxes[2].animate.set_color(INFLUENCE_COLOR).scale(1.1),
            rate_func=there_and_back,
            run_time=1
        )
        self.play(
            boxes[1].animate.set_color(WHITE).scale(1.2),
            rate_func=there_and_back,
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "An averaging kernel smooths out noisy, jittery data."
        self.play(self.lecture[2].animate.set_color(SMOOTH_COLOR))
        
        # Conceptual smoothing: highlight the whole kernel as a single unit
        highlight_rect = SurroundingRectangle(kernel_group, color=SMOOTH_COLOR, buff=0.2)
        self.play(Create(highlight_rect))
        self.play(Indicate(kernel_group, color=SMOOTH_COLOR))
        self.play(FadeOut(highlight_rect))
        
        self.wait(2)
