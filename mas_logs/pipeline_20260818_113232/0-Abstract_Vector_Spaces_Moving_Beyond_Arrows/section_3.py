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
        lecture_lines = ["Consider the set of polynomials.", "Addition follows algebraic rules.", "Scalar multiplication remains defined."]
        self.setup_layout("Example: The Space of Polynomials", lecture_lines)
        
        # Define objects once
        poly1 = MathTex("f(x) = ax^2 + bx + c", color=WHITE)
        poly_sum = MathTex("f(x) + g(x)", color="#FFD700")
        result = MathTex("h(x) = p(x)", color="#00FFFF")
        group_label = Text("Set of all Polynomials", color="#FF4500", font_size=24)
        
        # Placeholder assets
        icon1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.place_at_grid(poly1, 'B2', scale_factor=0.8)
        self.place_at_grid(icon1, 'B5', scale_factor=0.3)
        self.play(Write(poly1), FadeIn(icon1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.place_at_grid(poly_sum, 'C2', scale_factor=0.8)
        self.play(FadeIn(poly_sum))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.place_at_grid(result, 'D2', scale_factor=0.8)
        self.play(TransformFromCopy(poly_sum, result))
        self.wait(1)
        
        # Additional visuals for space concept
        self.place_in_area(group_label, 'E2', 'E5', scale_factor=0.8)
        self.place_at_grid(icon2, 'E6', scale_factor=0.3)
        self.play(Write(group_label), FadeIn(icon2))
        self.play(Flash(group_label, color="#FF4500"))
        self.wait(2)
