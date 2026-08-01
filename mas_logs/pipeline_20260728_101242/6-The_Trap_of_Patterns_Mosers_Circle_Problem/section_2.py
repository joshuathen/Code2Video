from manim import *

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
        # Setup layout with section-specific title and lecture lines
        self.setup_layout(
            "Prerequisite: Intersections and Euler’s Formula",
            [
                "Intersecting lines create new vertices and split edges.",
                "Euler's formula connects vertices, edges, and regions.",
                "Every inner intersection creates a brand new region."
            ]
        )
        
        # Colors:
        # Line 1: Yellow (#FFFF00) -> Chords
        # Line 2: Cyan (#00FFFF) -> Formula
        # Line 3: Red (#FF0000) -> Intersection Point

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line and draw circle/chords
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Issue 40 fix: Move circle to C2-F5 area
        circle = Circle(radius=1.3, color="#FFFFFF")
        self.place_in_area(circle, 'C2', 'F5', scale_factor=1.0)
        
        # Define chords inside the circle
        # We need them to intersect at the center for clarity in this prerequisite stage
        chord1 = Line(
            circle.point_at_angle(150 * DEGREES), 
            circle.point_at_angle(-30 * DEGREES), 
            color="#FFFF00"
        )
        chord2 = Line(
            circle.point_at_angle(210 * DEGREES), 
            circle.point_at_angle(30 * DEGREES), 
            color="#FFFF00"
        )
        
        self.play(Create(circle))
        self.play(Create(chord1), Create(chord2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line and show the formula
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # Euler's Formula V - E + F = 2
        formula = MathTex("V - E + F = 2", color="#00FFFF")
        # Issue 39 fix: Position formula in B2-B5 area
        self.place_in_area(formula, 'B2', 'B5', scale_factor=1.2)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line and emphasize intersection
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        
        # Intersection point dot and label at the center of the chords/circle
        center_point = circle.get_center()
        v_dot = Dot(center_point, color="#FF0000")
        v_label = Text("V", color="#FF0000", font_size=24)
        v_label.next_to(v_dot, UR, buff=0.1)
        
        self.play(FadeIn(v_dot), Write(v_label))
        self.play(Indicate(v_dot))
        self.wait(2)
