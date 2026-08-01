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
        # Section title and lecture lines as provided
        title = "Prerequisite: The Complex Playground"
        lecture_lines = [
            "We start in the complex plane, our mathematical playground.",
            "A point at one represents our starting real value.",
            "Multiplying by i rotates the point ninety degrees."
        ]
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Matching color: WHITE for axis line and first lecture point
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Real axis (D1 to D6) - WHITE
        real_axis = Line(self.grid["D1"], self.grid["D6"], color=WHITE)
        # Imaginary axis (F3 to A3) - CYAN
        imag_axis = Line(self.grid["F3"], self.grid["A3"], color="#00FFFF")
        
        # Axis labels
        re_label = Text("Real", font_size=22, color=WHITE)
        self.place_at_grid(re_label, "C6")
        
        im_label = Text("Imaginary", font_size=22, color="#00FFFF")
        # ISSUE 28 FIX: place at B3, scale 0.7
        self.place_at_grid(im_label, "B3", scale_factor=0.7)
        
        self.play(Create(real_axis), Create(imag_axis))
        self.play(Write(re_label), Write(im_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Matching color: YELLOW for the i-Bot / point
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Point 1 at D4 (1 unit right of D3)
        ibot = Dot(point=self.grid["D4"], color=YELLOW, radius=0.15)
        label_1 = Text("1", font_size=24, color=YELLOW)
        self.place_at_grid(label_1, "E4")
        
        self.play(FadeIn(ibot), Write(label_1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Matching color: PINK for the primary rotation
        self.play(self.lecture[2].animate.set_color(PINK))
        
        # Rotation 1: 1 (D4) -> i (C3)
        arc1 = Arc(radius=1.0, start_angle=0, angle=PI/2, arc_center=self.grid["D3"], color=PINK)
        label_i = Text("i", font_size=24, color=PINK)
        # ISSUE 29 FIX: place at B2, scale 0.8 to avoid axis label overlap
        self.place_at_grid(label_i, "B2", scale_factor=0.8)
        
        self.play(
            MoveAlongPath(ibot, arc1),
            ReplacementTransform(label_1, label_i),
            run_time=1.5
        )
        self.wait(1)

        # Demonstration of subsequent rotations
        self.play(self.lecture[2].animate.set_color(ORANGE))
        
        # Rotation 2: i (C3) -> -1 (D2)
        arc2 = Arc(radius=1.0, start_angle=PI/2, angle=PI/2, arc_center=self.grid["D3"], color=ORANGE)
        label_neg1 = Text("-1", font_size=24, color=ORANGE)
        self.place_at_grid(label_neg1, "E2")
        self.play(MoveAlongPath(ibot, arc2), ReplacementTransform(label_i, label_neg1), run_time=1)
        
        # Rotation 3: -1 (D2) -> -i (E3)
        arc3 = Arc(radius=1.0, start_angle=PI, angle=PI/2, arc_center=self.grid["D3"], color=ORANGE)
        label_negi = Text("-i", font_size=24, color=ORANGE)
        # ISSUE 30 FIX: place at F2, scale 0.8
        self.place_at_grid(label_negi, "F2", scale_factor=0.8)
        self.play(MoveAlongPath(ibot, arc3), ReplacementTransform(label_neg1, label_negi), run_time=1)
        
        # Rotation 4: -i (E3) -> 1 (D4)
        arc4 = Arc(radius=1.0, start_angle=3*PI/2, angle=PI/2, arc_center=self.grid["D3"], color=ORANGE)
        label_1_final = Text("1", font_size=24, color=ORANGE)
        self.place_at_grid(label_1_final, "E4")
        self.play(MoveAlongPath(ibot, arc4), ReplacementTransform(label_negi, label_1_final), run_time=1)
        
        self.wait(2)
