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
        # Data from shared state
        title_str = "Prerequisite: The Vector as a Navigation Instruction"
        lecture_lines = [
            "Imagine a vector as a navigation instruction.",
            "We start on a standard 2D coordinate grid.",
            "Every journey begins at the origin point.",
            "This instruction moves three right and two up.",
            "A single arrow represents this specific movement."
        ]
        
        self.setup_layout(title_str, lecture_lines)
        
        # Initial state: Hide everything for storyboard-accurate entry
        self.title.set_opacity(0)
        for line in self.lecture:
            line.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        # Background is #000000. Display a 2D coordinate grid (#555555).
        # Highlight lecture line 1 by changing its color to #00CCFF.
        # Fade in the title "The Vector as Navigation" (#FFFFFF).
        
        plane = NumberPlane(
            x_range=[-1, 3, 1],
            y_range=[-1, 3, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_color": "#555555"},
            axis_config={"stroke_color": "#555555"}
        )
        # Positioned in the area B2-F6
        self.place_in_area(plane, 'B2', 'F6')
        
        self.play(
            FadeIn(plane),
            self.title.animate.set_opacity(1),
            self.lecture[0].animate.set_opacity(1).set_color("#00CCFF"),
            run_time=1.0
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2 (#00CCFF). Show a vector [3, 2] represented
        # as a set of numeric coordinates near the origin.
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_opacity(1).set_color("#00CCFF"),
            run_time=0.5
        )
        
        # Numeric coordinates: Fix Issue 17 by moving to B3 and scaling to 0.7
        vec_coords = MathTex(r"\begin{bmatrix} 3 \\ 2 \end{bmatrix}", color="#00CCFF")
        self.place_at_grid(vec_coords, 'B3', scale_factor=0.7)
        
        self.play(Write(vec_coords), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3 (#00CCFF). Pulse a yellow dot (#FFFF00) at (0,0) labeled "Origin".
        
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_opacity(1).set_color("#00CCFF"),
            run_time=0.5
        )
        
        # Origin dot at E3 (which is roughly (0,0) in our plane's placement)
        origin_dot = Dot(color="#FFFF00")
        self.place_at_grid(origin_dot, 'E3')
        
        origin_label = Text("Origin", font_size=16, color="#FFFF00")
        self.place_at_grid(origin_label, 'F3', scale_factor=0.8)
        
        self.play(FadeIn(origin_dot), FadeIn(origin_label), run_time=0.5)
        self.play(origin_dot.animate.scale(1.5), run_time=0.3)
        self.play(origin_dot.animate.scale(1/1.5), run_time=0.3)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight lecture line 4 (#00CCFF). Draw blue dashed line (#00CCFF) from (0,0) to (3,0),
        # then a green dashed line (#33FF33) up to (3,2).
        
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_opacity(1).set_color("#00CCFF"),
            run_time=0.5
        )
        
        # Dashed lines: (0,0)->E3, (3,0)->E6, (3,2)->C6
        x_dash = DashedLine(self.grid['E3'], self.grid['E6'], color="#00CCFF")
        y_dash = DashedLine(self.grid['E6'], self.grid['C6'], color="#33FF33")
        
        self.play(Create(x_dash), run_time=1)
        self.play(Create(y_dash), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight lecture line 5 (#00CCFF). Draw a thick magenta arrow (#FF3399) 
        # from (0,0) to (3,2). Label it "[3, 2]".
        
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_opacity(1).set_color("#00CCFF"),
            run_time=0.5
        )
        
        arrow = Arrow(
            self.grid['E3'], 
            self.grid['C6'], 
            buff=0, 
            stroke_width=8, 
            color="#FF3399",
            max_tip_length_to_length_ratio=0.15
        )
        
        # Arrow label: Fix Issue 18 by scaling to 0.7
        arrow_label = MathTex(r"\begin{bmatrix} 3 \\ 2 \end{bmatrix}", color="#FF3399")
        self.place_at_grid(arrow_label, 'B6', scale_factor=0.7)
        
        self.play(
            Create(arrow),
            FadeIn(arrow_label),
            FadeOut(vec_coords), # Clean up temporary coordinates
            run_time=1.5
        )
        
        self.wait(3)
