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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite Review: The Exponential Foundation", [
            "Powers represent base growth over time.",
            "We visualize this as a growing tree.",
            "Base 'a' times time 'b' is result 'c'."
        ])
        
        # --- Create Elements ---
        eq = MathTex("a", "^", "b", "=", "c", font_size=60)
        eq.set_color(WHITE)
        
        # Load asset
        tree = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tree.svg")
        
        # Visuals for issue fixes
        self.place_in_area(eq, 'B2', 'D4', scale_factor=1.2)
        self.place_in_area(tree, 'A2', 'F5', scale_factor=0.9)
        
        # Bounding box around formula
        box = SurroundingRectangle(eq, color=BLUE, buff=0.3)

        # === Animation for Lecture Line 1 ===
        # Fade in base equation and tree
        self.play(FadeIn(eq), FadeIn(tree))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        # Highlight components
        self.play(
            eq[0].animate.set_color("#00FFFF"), # a (base)
            eq[2].animate.set_color("#FFFF00"), # b (exponent)
            eq[4].animate.set_color("#FF00FF"), # c (result)
            self.lecture[1].animate.set_color("#FFFFFF")
        )

        # === Animation for Lecture Line 3 ===
        # Animate box around equation
        self.play(Create(box), self.lecture[2].animate.set_color("#FFFFFF"))
        
        self.wait(2)
