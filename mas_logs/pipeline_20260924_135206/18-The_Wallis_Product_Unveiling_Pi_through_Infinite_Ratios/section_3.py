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
        lecture_lines = ["Evaluate the ratio I_{2n+1} over I_{2n}.", "As n grows, this ratio approaches one.", "Balance the integral products to isolate pi.", "The tug-of-war reveals pi/2.", "Algebraic manipulation bridges integrals to pi."]
        self.setup_layout("Deriving the Ratio", lecture_lines)
        
        # Elements
        ratio_expr = MathTex(r"{{I_{2n+1}}} \over {{I_{2n}}}", color=WHITE)
        scale_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg")
        rope_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rope.svg")
        
        term_rel = MathTex(r"\text{Relationship: } n \to \infty \implies 1", color="#FF33FF")
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(scale_icon, 'A2', scale_factor=0.5)
        self.play(FadeIn(self.place_at_grid(ratio_expr, 'B4', scale_factor=1.2)), FadeIn(scale_icon))
        self.lecture[0].set_color("#3357FF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF33FF")
        self.play(FadeIn(self.place_at_grid(term_rel, 'D3', scale_factor=0.8)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        balance_label = Text("Tug-of-War Balance", font_size=20, color=YELLOW)
        self.place_at_grid(rope_icon, 'A5', scale_factor=0.5)
        self.play(FadeIn(self.place_at_grid(balance_label, 'B5', scale_factor=0.9)), FadeIn(rope_icon))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(ORANGE)
        res_pi = MathTex(r"\pi/2", color=ORANGE)
        self.play(FadeIn(self.place_at_grid(res_pi, 'D5', scale_factor=1.0)))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GREEN)
        self.wait(1)
