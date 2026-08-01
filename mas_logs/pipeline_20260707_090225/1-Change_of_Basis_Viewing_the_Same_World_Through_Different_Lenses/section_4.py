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
        # Title and Lecture Lines
        self.setup_layout(
            "The Core Concept: One Vector, Two Names",
            [
                "A physical vector is a fixed arrow in space.",
                "In our standard grid, we call it zero, three.",
                "But the slanted grid offers a different perspective.",
                "We reach the same tip using one b1, two b2.",
                "This same vector is named one, two in Basis B."
            ]
        )

        # Positioning Helpers
        origin = self.grid["E2"]
        tip = self.grid["B2"]
        
        # Basis Vectors in Screen Space
        # Basis B Step
        b1_step = self.grid["D1"] - self.grid["E2"]  # Approximately [-1, 1, 0]
        b2_step = (self.grid["B2"] - self.grid["D1"]) / 2  # Approximately [0.5, 1, 0]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        physical_vector = Arrow(start=origin, end=tip, buff=0, color=WHITE, stroke_width=6)
        
        # Asset: ground.svg
        ground = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/ground.svg")
        self.place_in_area(ground, "F1", "F6", scale_factor=0.6)
        ground.set_color(GRAY_E)
        
        circle = Circle(radius=0.1, color=WHITE, fill_opacity=0.5)
        self.place_at_grid(circle, "B2", scale_factor=0.8)
        
        self.play(
            FadeIn(ground),
            GrowArrow(physical_vector),
            Create(circle),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE_C)
        
        # Standard Grid
        std_grid = VGroup()
        for i in range(6):
            h_line = Line(self.grid[f"{chr(65+i)}1"], self.grid[f"{chr(65+i)}6"], color="#666666", stroke_opacity=0.4)
            v_line = Line(self.grid[f"A{i+1}"], self.grid[f"F{i+1}"], color="#666666", stroke_opacity=0.4)
            std_grid.add(h_line, v_line)
        
        # Fixed: Replaced MathTex with Text to avoid LaTeX dependency
        label_std = Text("[0, 3] std", color=BLUE_C, font_size=28)
        label_std.next_to(tip, RIGHT, buff=0.2)
        
        square = Square(side_length=0.2, color=BLUE_C, fill_opacity=0.2)
        self.place_at_grid(square, "C3", scale_factor=0.8)

        self.play(
            Create(std_grid),
            Write(label_std),
            Create(square),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN_B)
        
        # Slanted Grid
        slanted_grid = VGroup()
        for i in range(-3, 4):
            start_b2 = origin + i * b1_step - 2 * b2_step
            end_b2 = origin + i * b1_step + 4 * b2_step
            line_b2 = Line(start_b2, end_b2, color=GREEN_B, stroke_opacity=0.3)
            start_b1 = origin + i * b2_step - 2 * b1_step
            end_b1 = origin + i * b2_step + 2 * b1_step
            line_b1 = Line(start_b1, end_b1, color=GREEN_B, stroke_opacity=0.3)
            slanted_grid.add(line_b1, line_b2)

        self.play(
            std_grid.animate.set_stroke(opacity=0.1),
            FadeIn(slanted_grid),
            FadeOut(label_std),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW_B)
        
        path_b1 = Arrow(start=origin, end=origin + b1_step, buff=0, color=YELLOW_B, stroke_width=4)
        path_b2_1 = Arrow(start=origin + b1_step, end=origin + b1_step + b2_step, buff=0, color=GOLD_B, stroke_width=4)
        path_b2_2 = Arrow(start=origin + b1_step + b2_step, end=origin + b1_step + 2*b2_step, buff=0, color=GOLD_B, stroke_width=4)
        
        triangle = Triangle().scale(0.1).set_color(YELLOW_B)
        self.place_at_grid(triangle, "D4", scale_factor=0.8)

        self.play(
            GrowArrow(path_b1),
            Create(triangle),
            run_time=1
        )
        self.play(
            GrowArrow(path_b2_1),
            run_time=0.8
        )
        self.play(
            GrowArrow(path_b2_2),
            run_time=0.8
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GOLD_A)
        
        # Fixed: Replaced MathTex with Text to avoid LaTeX dependency
        label_basis = Text("[1, 2] B", color=GOLD_A, font_size=32)
        label_basis.next_to(tip, UP, buff=0.2)
        
        self.play(
            Write(label_basis),
            physical_vector.animate.set_color(GOLD_A),
            run_time=1.5
        )
        self.wait(3)
