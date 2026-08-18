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
        lecture_lines = [
            "Independence simplifies Bayes' Theorem significantly.",
            "The formula collapses when events are independent.",
            "Independent evidence provides zero information."
        ]
        self.setup_layout("Bayes and Independence: The Critical Link", lecture_lines)
        
        # Asset loader
        def load_asset(path):
            try:
                return SVGMobject(path)
            except:
                return Circle(radius=0.2, color=GRAY) # Fallback

        # === Animation for Lecture Line 1 ===
        bayes_formula = MathTex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}", color="#FFFFFF")
        asset1 = load_asset("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_in_area(bayes_formula, 'A2', 'C4', scale_factor=1.0)
        self.place_at_grid(asset1, 'A5', scale_factor=0.5)
        
        self.play(Write(bayes_formula), FadeIn(asset1))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        independence_text = MathTex(r"P(A|B) = P(A)", color="#FF5555")
        self.place_at_grid(independence_text, 'E3', scale_factor=1.0)
        self.play(FadeIn(independence_text))
        self.play(self.lecture[1].animate.set_color("#FF5555"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        asset2 = load_asset("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.play(FadeOut(bayes_formula), FadeOut(independence_text), FadeOut(asset1))
        
        zero_info = Text("Independence = Zero Information", color="#FFFFFF")
        self.place_in_area(zero_info, 'D2', 'D5', scale_factor=0.9)
        self.place_at_grid(asset2, 'C3', scale_factor=0.5)
        
        self.play(Write(zero_info), FadeIn(asset2))
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.wait(2)
