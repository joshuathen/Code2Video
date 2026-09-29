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
        self.setup_layout("Formal Definition: Av = λv", [
            "The equation is A times v equals lambda v.",
            "A is the matrix, v is the eigenvector.",
            "Lambda is the scalar eigenvalue factor.",
            "If lambda exceeds one, the vector stretches.",
            "A negative lambda flips the vector's direction."
        ])
        
        # Define objects
        eq = MathTex(r"A", r"v", "=", r"\lambda", r"v", font_size=42)
        eq.set_color_by_tex("A", BLUE)
        eq.set_color_by_tex("v", GREEN)
        eq.set_color_by_tex(r"\lambda", YELLOW)
        
        # Fixing Equation positioning (Issue 20, 32)
        self.place_in_area(eq, 'B3', 'B4', scale_factor=0.9)

        # Assets
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        vector_v = Vector([0, 1.5], color=GREEN)
        scaled_v = Vector([0, 2.5], color=YELLOW)
        flipped_v = Vector([0, -1.5], color=RED)
        
        animation_group = VGroup(asset_icon, vector_v, scaled_v, flipped_v)
        
        # Fixing Animation positioning (Issue 21, 33)
        self.place_at_grid(animation_group, 'C2', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(Write(eq))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        self.play(FadeIn(asset_icon), FadeIn(vector_v))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(ORANGE)
        self.play(ReplacementTransform(vector_v.copy(), scaled_v))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(RED)
        self.play(ReplacementTransform(vector_v.copy(), flipped_v))
        self.wait(2)
