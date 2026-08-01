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
        # Initialize layout
        self.setup_layout(
            "Prerequisite 1: The Secret Power of 'i'", 
            [
                "Let's look at i beyond just a square root.", 
                "Multiplication by i acts as a ninety-degree turn.", 
                "It rotates numbers from horizontal to vertical."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Color highlighting (instant to avoid previous play error)
        self.lecture[0].set_color(WHITE)
        
        # Define the origin and axes for the complex plane manually using grid
        # Center of rotation/origin at D4
        origin_point = self.grid["D4"]
        
        # Draw a simple axis system centered at D4
        # Horizontal: D1 to D6
        h_line = Line(self.grid["D1"], self.grid["D6"], color=GREY_C, stroke_width=2)
        # Vertical: F4 to A4
        v_line = Line(self.grid["F4"], self.grid["A4"], color=GREY_C, stroke_width=2)
        plane = VGroup(h_line, v_line)
        
        # Create a circle as the 'explorer' (white dot) at coordinate (1,0) -> Grid D5
        circle = Circle(radius=0.15, color=WHITE, fill_opacity=1)
        self.place_at_grid(circle, "D5", scale_factor=0.8)
        
        # Label '1' at D5
        label_1 = Text("1", font_size=32, color=WHITE)
        label_1.next_to(circle, DOWN, buff=0.2)
        
        self.play(Create(plane), FadeIn(circle), Write(label_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fix: Using .animate to avoid 'TypeError: Unexpected argument VMobjectFromSVGPath'
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Vector from origin (D4) to (1,0) (D5)
        vec = Arrow(start=origin_point, end=self.grid["D5"], buff=0, color=WHITE, stroke_width=4)
        
        # Rotation animation: 90 degrees CCW
        # The rotation target (0,1) is grid C4.
        self.play(Create(vec))
        self.play(
            Rotate(vec, angle=PI/2, about_point=origin_point, rate_func=smooth),
            circle.animate.move_to(self.grid["C4"]),
            vec.animate.set_color("#00FFFF"), # Turning cyan
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fix: Using .animate to avoid 'TypeError: Unexpected argument VMobjectFromSVGPath'
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        # Square acting as a background for label at B3 (Addressing Issues 36 and 37)
        bg_square = Square(side_length=0.8, color=YELLOW, fill_opacity=0.2)
        self.place_at_grid(bg_square, "B3", scale_factor=0.7)
        
        # New label 'i' at C4
        label_i = Text("i", font_size=36, color="#00FFFF", slant=ITALIC)
        label_i.next_to(circle, LEFT, buff=0.2)
        
        # 90 Rotation text at B3
        rot_text = Text("90° Rotation", font_size=20, color=WHITE)
        # Using move_to instead of place_at_grid because bg_square is already at B3
        rot_text.move_to(bg_square.get_center())
        
        self.play(
            FadeIn(bg_square),
            Write(rot_text),
            ReplacementTransform(label_1, label_i)
        )
        self.wait(3)
