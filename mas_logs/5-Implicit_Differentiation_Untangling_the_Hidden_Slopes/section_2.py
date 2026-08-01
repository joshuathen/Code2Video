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
        # Initialize the layout with the title and the script provided.
        self.setup_layout("Prerequisite Check: The Chain Rule 'Sidecar'", [
            "Since y depends on x, we use the Chain Rule.",
            "Differentiating x^3 gives 3x^2 simply.",
            "But differentiating y^3 requires a \"sidecar\" term.",
            "We get 3y^2 times the derivative of y, dy/dx.",
            "Always attach dy/dx when differentiating terms with y."
        ])

        # === Animation for Lecture Line 1 ===
        # "Since y depends on x, we use the Chain Rule."
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # "Differentiating x^3 gives 3x^2 simply."
        # Action: Display 'd/dx [x^3] = 3x^2' below (Row C)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        x_example = Text("d/dx[x^3] = 3x^2", color=WHITE)
        self.place_in_area(x_example, "C1", "C6", scale_factor=1.2)
        self.play(Write(x_example))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # "But differentiating y^3 requires a \"sidecar\" term."
        # Action: Display 'd/dx [y^3]' in white (#FFFFFF) (Row A)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        y_lhs = Text("d/dx[y^3]", color="#FFFFFF")
        self.place_at_grid(y_lhs, "A2", scale_factor=1.2)
        self.play(Write(y_lhs))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # "We get 3y^2 times the derivative of y, dy/dx."
        # Action: Show '3y^2' appearing, then attach '(dy/dx)' in orange (#FF8800)
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        eq_sign = Text("=", color=WHITE)
        self.place_at_grid(eq_sign, "A3", scale_factor=1.2)
        
        y_deriv_base = Text("3y^2", color=WHITE)
        self.place_at_grid(y_deriv_base, "A4", scale_factor=1.2)
        
        sidecar = Text("· dy/dx", color="#FF8800")
        self.place_at_grid(sidecar, "A5", scale_factor=1.2)
        
        self.play(Write(eq_sign), Write(y_deriv_base))
        self.play(FadeIn(sidecar, shift=RIGHT))
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # "Always attach dy/dx when differentiating terms with y."
        # Action: Highlight (dy/dx) with a box and contrast with the x-case (no sidecar)
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        sidecar_box = SurroundingRectangle(sidecar, color="#FF8800", buff=0.1)
        sidecar_label = Text("Sidecar", font_size=24, color="#FF8800")
        self.place_at_grid(sidecar_label, "B5", scale_factor=0.8)
        
        no_sidecar_note = Text("(no dy/dx needed)", font_size=20, color=GREY_A)
        self.place_at_grid(no_sidecar_note, "D4", scale_factor=0.8)
        
        self.play(Create(sidecar_box), Write(sidecar_label))
        self.play(Write(no_sidecar_note))
        self.wait(3)