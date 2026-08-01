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
        title_str = "The Shortcut: Diagonalization"
        lecture_lines = [
            "Computing infinite sums of matrices is often difficult.",
            "Diagonalization provides a powerful shortcut for calculations.",
            "Perspective changes make the system's components independent.",
            "Exponentiating a diagonal matrix is fast and simple.",
            "Finally, transform back to the original coordinate system."
        ]
        
        self.setup_layout(title_str, lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        
        exp_series = Text(
            "eᴬ = I + A + A²/2! + A³/3! + ...",
            font_size=36,
            slant=ITALIC
        )
        self.place_in_area(exp_series, 'A1', 'B6', scale_factor=0.8)
        
        self.play(Write(exp_series))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#50C878") # Emerald Green
        )
        
        try:
            shortcut_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/shortcut.svg")
            shortcut_icon.set_color("#50C878")
        except:
            shortcut_icon = Circle(radius=0.3, color="#50C878")
        
        diag_formula = Text("A = PDP⁻¹", color="#50C878", font_size=40, slant=ITALIC)
        
        diag_text = VGroup(shortcut_icon, diag_formula).arrange(RIGHT, buff=0.5)
        self.place_in_area(diag_text, 'C1', 'C6', scale_factor=0.7)
        
        self.play(FadeIn(shortcut_icon), Write(diag_formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(BLUE)
        )
        
        # Highlight 'P' - Coordinate transformation matrix
        p_box = SurroundingRectangle(diag_formula[4], color=WHITE, buff=0.1) 
        self.play(Create(p_box))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#50C878")
        )
        
        # Highlight 'D' - Eigenvalue matrix
        d_box = SurroundingRectangle(diag_formula[5], color="#50C878", buff=0.1)
        self.play(ReplacementTransform(p_box, d_box))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(BLUE)
        )
        
        # Highlight 'P⁻¹' - Reverse transformation
        pinv_box = SurroundingRectangle(diag_formula[6:], color=WHITE, buff=0.1)
        self.play(ReplacementTransform(d_box, pinv_box))
        self.wait(2)
