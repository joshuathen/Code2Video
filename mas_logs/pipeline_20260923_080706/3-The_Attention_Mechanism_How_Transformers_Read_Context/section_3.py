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
        self.setup_layout("Calculating Attention Scores", [
            "Dot-product measures similarity between vectors.",
            "Softmax converts scores to probability weights.",
            "Weights sum to one across context."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Q * K^T visualization
        matrix_q = Matrix([["q1"], ["q2"]], left_bracket="[", right_bracket="]")
        matrix_k = Matrix([["k1", "k2"]], left_bracket="[", right_bracket="]")
        eq = MathTex(r"Q \cdot K^T = \text{Score}")
        
        icon_vec = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg")
        
        self.place_at_grid(matrix_q, "B2", scale_factor=0.8)
        self.place_at_grid(matrix_k, "B4", scale_factor=0.8)
        self.place_in_area(eq, "D2", "D4", scale_factor=1.0)
        self.place_at_grid(icon_vec, "A3", scale_factor=0.5)
        
        self.play(FadeIn(matrix_q), FadeIn(matrix_k), FadeIn(eq), FadeIn(icon_vec))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Heatmap intensity (simulated with colored squares)
        heatmap = VGroup(*[Square(side_length=0.6, fill_opacity=0.6, color=RED).shift(i*0.7*RIGHT + j*0.7*UP) for i in range(2) for j in range(2)])
        softmax_text = MathTex(r"\text{Softmax}(\text{Score})")
        icon_bar = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bar.svg")
        
        self.place_in_area(heatmap, "B5", "C6", scale_factor=0.9)
        self.place_at_grid(softmax_text, "D5", scale_factor=0.7)
        self.place_at_grid(icon_bar, "C5", scale_factor=0.5)
        
        self.play(FadeIn(heatmap), FadeIn(softmax_text), FadeIn(icon_bar))
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Weights sum to 1
        sum_text = MathTex(r"\sum w_i = 1")
        icon_grad = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gradient.svg")
        
        self.place_at_grid(sum_text, "E3", scale_factor=0.8)
        self.place_at_grid(icon_grad, "F3", scale_factor=0.5)
        
        self.play(Write(sum_text), FadeIn(icon_grad))
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.wait(2)
