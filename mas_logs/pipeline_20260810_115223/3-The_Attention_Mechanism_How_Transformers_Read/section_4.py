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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Mathematical Intuition: Scaled Dot-Product", [
            "- QK transpose measures similarity between words.",
            "- Scale by root dk for stability.",
            "- Softmax normalizes scores into probabilities."
        ])
        
        # Main formula
        formula = MathTex(
            "\\\\text{Attention}(Q, K, V) = \\\\text{softmax}\\\\left(",  # 0
            "QK^T",                                                  # 1
            "/\\\\sqrt{d_k}",                                          # 2
            "\\\\right)V",                                             # 3
            font_size=36
        )
        
        # Load assets
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        scale_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg")
        ruler_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")

        # === Animation for Lecture Line 1 ===
        self.place_in_area(formula, 'A2', 'B5', scale_factor=0.85)
        self.play(FadeIn(formula))
        
        self.place_at_grid(calc_icon, 'C2', scale_factor=0.6)
        self.play(FadeIn(calc_icon), formula[1].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(scale_icon, 'C4', scale_factor=0.6)
        self.play(self.lecture[0].animate.set_color(WHITE),
                  self.lecture[1].animate.set_color("#00FFFF"),
                  FadeIn(scale_icon), 
                  formula[2].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(ruler_icon, 'C5', scale_factor=0.6)
        self.play(self.lecture[1].animate.set_color(WHITE),
                  self.lecture[2].animate.set_color("#FFFF00"),
                  FadeIn(ruler_icon),
                  formula[0].animate.set_color("#FFFF00"), 
                  formula[3].animate.set_color("#FFFF00"))
        
        self.wait(2)
