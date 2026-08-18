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
        self.setup_layout("Fractional Exponents", [
            "Roots are exponents in fraction form.",
            "The square root of 9 is 9 to the 1/2.",
            "Like a tree branching back to its original base."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Fade in 'b^(1/n) = n√b' in white with tree asset
        tree_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tree.svg")
        eq1 = MathTex(r"b^{1/n} = \sqrt[n]{b}", color=WHITE)
        self.place_at_grid(eq1, 'B4', scale_factor=1.2)
        self.place_at_grid(tree_asset, 'B6', scale_factor=0.5)
        self.play(FadeIn(eq1), FadeIn(tree_asset))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Show '9^(1/2) = √9' example in cyan
        # Flash the equivalence symbol '=' in red
        eq2 = MathTex(r"9^{1/2} = \sqrt{9}", color=TEAL)
        eq2_eq_sign = eq2.get_part_by_tex("=")
        self.place_at_grid(eq2, 'C2', scale_factor=1.5)
        
        self.play(FadeIn(eq2))
        self.play(Flash(eq2_eq_sign, color=RED, line_length=0.2, flash_radius=0.3))
        self.lecture[1].set_color(TEAL)

        # === Animation for Lecture Line 3 ===
        # Visual representation of growth with tree asset
        tree = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tree.svg")
        self.place_at_grid(tree, 'D4', scale_factor=1.0)
        self.play(Create(tree))
        self.lecture[2].set_color(GREEN)
        
        self.wait(2)
