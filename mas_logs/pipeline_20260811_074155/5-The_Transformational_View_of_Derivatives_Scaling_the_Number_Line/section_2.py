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
        title = "Prerequisite: Linear Scaling (The Constant Stretch)"
        lecture_lines = [
            "Consider a simple linear function like f(x) = 3x.",
            "Any interval is stretched by a factor of three.",
            "This constant scaling factor is the function's derivative."
        ]
        self.setup_layout(title, lecture_lines)

        # Assets
        stretch_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/stretch.svg"
        factor_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/factor.svg"

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Formula f(x) = 3x
        # Issue 23: Centering formula at A3-A4
        formula = MathTex("f(x) = 3x", color="#FFFF00")
        self.place_in_area(formula, 'A3', 'A4', scale_factor=1.2)
        
        # Input Number Line
        # Issue 24: Shift input_line to B2-B6 to avoid cramping
        input_line = NumberLine(
            x_range=[-1, 5, 1],
            length=4.0,
            color=WHITE,
            include_numbers=True,
            font_size=18
        )
        self.place_in_area(input_line, "B2", "B6")
        
        # Label placed using grid for precision
        input_label = Text("Input x", font_size=18, color=WHITE)
        self.place_at_grid(input_label, "B1")
        
        # Interval [0, 1] on Input Line
        input_interval_rect = Line(
            input_line.n2p(0),
            input_line.n2p(1),
            color="#FFFF00",
            stroke_width=8
        )
        
        self.play(Create(input_line), Write(input_label))
        self.play(Write(formula))
        self.play(Create(input_interval_rect))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        
        # Output Number Line
        # Issue 25: Align output_line with input_line at E2-E6
        output_line = NumberLine(
            x_range=[-1, 15, 3], 
            length=4.0,
            color=WHITE,
            include_numbers=True,
            font_size=18
        )
        self.place_in_area(output_line, "E2", "E6")
        
        # Label placed using grid
        output_label = Text("Output f(x)", font_size=18, color=WHITE)
        self.place_at_grid(output_label, "E1")

        # Vertical dashed lines
        dash1 = DashedLine(input_line.n2p(0), output_line.n2p(0), color=GRAY)
        dash2 = DashedLine(input_line.n2p(1), output_line.n2p(3), color=GRAY)
        
        # Output interval [0, 3]
        output_interval_rect = Line(
            output_line.n2p(0),
            output_line.n2p(3),
            color="#FFFF00",
            stroke_width=8
        )

        # Load stretch asset
        # Issue 19: Integrate stretch.svg
        stretch_icon = SVGMobject(stretch_asset).set_height(0.4).set_color(WHITE)
        self.place_in_area(stretch_icon, "C3", "D4")

        self.play(Create(output_line), Write(output_label))
        self.play(Create(dash1), Create(dash2))
        
        # Animation: [0,1] segment stretching to cover [0,3]
        stretching_segment = input_interval_rect.copy()
        
        self.play(
            stretching_segment.animate.move_to(output_interval_rect.get_center()).stretch_to_fit_width(output_interval_rect.get_width()),
            FadeIn(stretch_icon),
            run_time=2
        )
        self.play(ReplacementTransform(stretching_segment, output_interval_rect), FadeOut(stretch_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FF00")
        )
        
        # Label 'Scaling Factor = 3'
        # Issue 19: Integrate factor.svg
        scaling_label = Text("Scaling Factor = 3", color="#00FF00", font_size=24)
        factor_icon = SVGMobject(factor_asset).set_height(0.6).set_color("#00FF00")
        
        # Group them for visual alignment
        label_group = VGroup(factor_icon, scaling_label).arrange(RIGHT, buff=0.2)
        self.place_in_area(label_group, "F2", "F5")
        
        self.play(FadeIn(label_group))
        self.wait(2)
