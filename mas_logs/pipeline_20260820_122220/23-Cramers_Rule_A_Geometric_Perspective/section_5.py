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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Limitations", [
            "Cramer's rule is an elegant geometric ratio.",
            "Expensive for large systems, efficient for small.",
            "Useful for intuitive understanding of systems."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display Cramer's Rule formula
        formula = MathTex(r"x_i = \frac{\det(A_i)}{\det(A)}", font_size=40, color="#FFFFFF")
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        
        # Anchor point columns 4-6
        self.place_at_grid(formula, "B4", scale_factor=1.2)
        calculator.next_to(formula, UP, buff=0.2)
        
        self.play(Write(formula), FadeIn(calculator))
        self.lecture[0].set_color("#00FF00")
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Show large 'X' mark over calculation for high-dim matrices
        large_x = Text("X", font_size=120, color="#FF0000")
        
        # Fix: Red 'X' on B4
        self.place_at_grid(large_x, "B4", scale_factor=0.8)
        
        self.play(FadeIn(large_x, scale=1.5))
        self.lecture[1].set_color("#FF0000")
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Fade to closing slide
        final_text = Text("Cramer's Rule: Intuition over Efficiency", font_size=28, color="#808080")
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        
        # Fix: Final text on C4
        self.place_at_grid(final_text, "C4", scale_factor=0.9)
        computer.next_to(final_text, DOWN, buff=0.2)
        
        self.play(FadeOut(formula), FadeOut(large_x), FadeOut(calculator), FadeIn(final_text), FadeIn(computer))
        self.lecture[2].set_color("#808080")
        self.wait(2)
