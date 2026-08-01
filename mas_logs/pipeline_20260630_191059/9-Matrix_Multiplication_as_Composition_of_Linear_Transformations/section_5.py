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
        # Metadata
        title = "Why Order Matters (Non-Commutativity)"
        lecture_lines = [
            "Let's compare two different sequences of transformations.",
            "Rotate then Shear creates one spatial result.",
            "Shearing then rotating produces a different outcome.",
            "Notice the distinct shapes Momo takes.",
            "This proves that matrix multiplication order matters."
        ]
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Let's compare two different sequences of transformations.
        self.lecture[0].set_color(WHITE)
        
        label_1 = Text("Scenario 1: Rotate -> Shear", font_size=18, color=BLUE)
        label_2 = Text("Scenario 2: Shear -> Rotate", font_size=18, color=YELLOW)
        
        # Positioning labels with better horizontal balance (Issue 36)
        self.place_in_area(label_1, "A1", "A3", scale_factor=0.8)
        self.place_in_area(label_2, "A4", "A6", scale_factor=0.8)
        
        # Using SVG asset (Issue 28)
        momo_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/m.svg"
        momo1 = SVGMobject(momo_path).set_color(BLUE)
        momo2 = SVGMobject(momo_path).set_color(YELLOW)
        
        # Positioning and scaling Momo (Issue 37)
        self.place_at_grid(momo1, "C2", scale_factor=0.6)
        self.place_at_grid(momo2, "C5", scale_factor=0.6)
        
        self.play(FadeIn(label_1), FadeIn(label_2), FadeIn(momo1), FadeIn(momo2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rotate then Shear creates one spatial result.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        
        # Matrix for shearing along X axis with factor 1
        shear_matrix = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        
        # Sequence 1: Rotate then Shear
        self.play(momo1.animate.rotate(90*DEGREES, about_point=momo1.get_center()), run_time=1.5)
        self.play(momo1.animate.apply_matrix(shear_matrix, about_point=momo1.get_center()), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Shearing then rotating produces a different outcome.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Sequence 2: Shear then Rotate
        self.play(momo2.animate.apply_matrix(shear_matrix, about_point=momo2.get_center()), run_time=1.5)
        self.play(momo2.animate.rotate(90*DEGREES, about_point=momo2.get_center()), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Notice the distinct shapes Momo takes.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(GREEN)
        
        # Highlight both final states
        box1 = SurroundingRectangle(momo1, color=GREEN, buff=0.1)
        box2 = SurroundingRectangle(momo2, color=GREEN, buff=0.1)
        
        self.play(Create(box1), Create(box2))
        self.wait(2)
        self.play(FadeOut(box1), FadeOut(box2))

        # === Animation for Lecture Line 5 ===
        # This proves that matrix multiplication order matters.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FF0000") # Red color for emphasis
        
        # Inequality text (Issue 38)
        inequality = Text("AB ≠ BA", font_size=48, color="#FF0000")
        self.place_in_area(inequality, "E2", "F5", scale_factor=1.2)
        
        self.play(
            FadeOut(momo1), FadeOut(momo2),
            FadeOut(label_1), FadeOut(label_2),
            Write(inequality)
        )
        
        # Flash the inequality to conclude
        self.play(Flash(inequality, color="#FF0000", line_length=0.4, flash_radius=1.2))
        self.wait(2)
