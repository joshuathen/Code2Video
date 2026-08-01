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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Let's see how these numbers reshape Leo's world.",
            "An eigenvalue of five expands space along one diagonal.",
            "An eigenvalue of two stretches the other diagonal.",
            "Values above one grow, while values below one shrink.",
            "Negative eigenvalues flip the orientation entirely."
        ]
        self.setup_layout("Visualizing the Resulting Transformation", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Grid Center for Area A1-F6
        area_center = (self.grid["A1"] + self.grid["F6"]) / 2
        
        # Create Leo the Cat (stylized)
        leo_body = Circle(radius=0.4, color="#FFFF00", fill_opacity=0.8)
        ear_l = Triangle(color="#FFFF00", fill_opacity=0.8).scale(0.15).rotate(30*DEGREES).move_to(leo_body.get_top()+LEFT*0.2+UP*0.05)
        ear_r = Triangle(color="#FFFF00", fill_opacity=0.8).scale(0.15).rotate(-30*DEGREES).move_to(leo_body.get_top()+RIGHT*0.2+UP*0.05)
        eye_l = Dot(radius=0.04, color=BLACK).move_to(leo_body.get_center() + LEFT*0.15 + UP*0.1)
        eye_r = Dot(radius=0.04, color=BLACK).move_to(leo_body.get_center() + RIGHT*0.15 + UP*0.1)
        leo = VGroup(leo_body, ear_l, ear_r, eye_l, eye_r)
        
        # Place Leo and Eigen-lines
        # ISSUE 40 FIX: scale factor reduced to 0.15 to avoid clipping on expansion
        self.place_in_area(leo, 'A1', 'F6', scale_factor=0.15)
        
        line1 = Line(area_center + DL*2, area_center + UR*2, color="#00FF00", stroke_width=2)
        line2 = Line(area_center + DR*2, area_center + UL*2, color="#00FF00", stroke_width=2)
        
        label1 = Text("\u03bb=5", color="#00FF00", font_size=20)
        label2 = Text("\u03bb=2", color="#00FF00", font_size=20)
        self.place_at_grid(label1, "B5")
        self.place_at_grid(label2, "B2")

        self.play(FadeIn(leo), Create(line1), Create(line2), FadeIn(label1), FadeIn(label2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Matrix to scale by factor 5 along 45 degree diagonal (Line1)
        # M = [[3, 2], [2, 3]]
        mat1 = [[3, 2], [2, 3]]
        self.play(leo.animate.apply_matrix(mat1, about_point=area_center), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Matrix to scale by factor 2 along 135 degree diagonal (Line2)
        # Given we already transformed, applying this matrix scales the other axis.
        # M = [[1.5, -0.5], [-0.5, 1.5]]
        mat2 = [[1.5, -0.5], [-0.5, 1.5]]
        self.play(leo.animate.apply_matrix(mat2, about_point=area_center), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Highlight transformation importance
        self.play(Flash(leo, color=WHITE, line_length=0.4))
        dna_text = Text("The DNA of the Transformation", color=WHITE, font_size=22)
        # ISSUE 39 FIX: place_in_area for better centering of long text
        self.place_in_area(dna_text, 'F1', 'F6', scale_factor=0.8)
        self.play(Write(dna_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Show a flip (reflection across Y axis relative to center)
        flip_mat = [[-1, 0], [0, 1]]
        self.play(leo.animate.apply_matrix(flip_mat, about_point=area_center), run_time=1.5)
        self.wait(2)
