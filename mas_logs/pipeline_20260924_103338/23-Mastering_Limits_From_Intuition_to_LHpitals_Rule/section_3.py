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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "L'Hôpital's Rule resolves these indeterminate forms easily.",
            "Compare the slopes of both functions at limits.",
            "The limit of the ratio equals derivative ratio.",
            "Tangent slopes bridge the gap to values.",
            "This shortcut simplifies complex limit evaluations."
        ]
        self.setup_layout("L'Hôpital's Rule: The Shortcut", lecture_lines)
        
        # Initial Expression
        original_ratio = MathTex(r"\lim_{x \to c} \frac{f(x)}{g(x)}", color=WHITE)
        label1 = Text("Original Ratio", font_size=20, color=WHITE)
        
        # Derivative Expression
        deriv_ratio = MathTex(r"\lim_{x \to c} \frac{f'(x)}{g'(x)}", color="#32CD32")
        label2 = Text("Derivative Ratio", font_size=20, color="#32CD32")
        
        # Equals sign
        equals = MathTex("=", color="#FFD700")
        label3 = Text("L'Hopital's Rule", font_size=20, color="#FFD700")
        
        # Target icon
        target_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg", color="#32CD32")

        # === Animation for Lecture Line 1 ===
        # Addressing Issue 22: Change area for original_ratio
        # Addressing Issue 23: Change area for label1
        self.place_in_area(original_ratio, 'B2', 'C3', scale_factor=1.2)
        self.place_at_grid(label1, 'B2', scale_factor=0.6)
        self.play(Write(original_ratio), FadeIn(label1))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#32CD32")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Addressing Issue 24: Fix derivative ratio grid space
        self.place_at_grid(equals, 'C4', scale_factor=1.5)
        self.place_in_area(deriv_ratio, 'C5', 'D6', scale_factor=1.2)
        self.place_at_grid(label2, 'B5', scale_factor=0.8)
        self.place_at_grid(label3, 'B4', scale_factor=0.8)
        
        self.play(
            Transform(original_ratio.copy(), deriv_ratio),
            Write(equals),
            Write(label2),
            Write(label3)
        )
        self.lecture[2].set_color("#FFD700")
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#32CD32")
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        # Addressing Issue 14: Use target asset
        self.place_at_grid(target_icon, 'B6', scale_factor=0.5)
        self.play(FadeIn(target_icon))
        self.lecture[4].set_color("#FFFFFF")
        self.wait(1)
