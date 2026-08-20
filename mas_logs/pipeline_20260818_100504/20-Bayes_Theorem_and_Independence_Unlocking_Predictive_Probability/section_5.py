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
        self.setup_layout("Synthesis & Key Takeaway", [
            "Independence means paths never interact.",
            "Bayes' Theorem creates a feedback loop.",
            "Use both to master predictive probability."
        ])
        
        # Prepare visual elements
        path_group = VGroup(
            Line(LEFT, RIGHT, color=BLUE),
            Line(LEFT, RIGHT, color=GREEN)
        ).arrange(DOWN, buff=0.5)
        
        loop = Circle(radius=0.7, color=YELLOW)
        loop_arrow = Arrow(UP, UP+0.01, color=YELLOW)
        
        formula = MathTex(r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}", font_size=36)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        # Fixed issue 33: adjusted position to D2-E5
        self.place_in_area(path_group, 'D2', 'E5', scale_factor=0.7)
        self.play(Create(path_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.play(FadeOut(path_group))
        self.place_in_area(VGroup(loop, loop_arrow), 'B3', 'D4')
        self.play(Create(loop), GrowArrow(loop_arrow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        self.play(FadeOut(loop), FadeOut(loop_arrow))
        
        # Fixed issues 32 and 34: use place_at_grid for cleaner positioning
        self.place_at_grid(formula, 'B4', scale_factor=0.5)
        
        # Highlight term P(B|A)
        formula_highlight = MathTex(r"P(B|A)", font_size=36, color="#FF00FF")
        formula_highlight.move_to(formula.get_parts_by_tex(r"P(B|A)"))
        
        self.play(Write(formula))
        self.play(ReplacementTransform(formula.get_parts_by_tex(r"P(B|A)").copy(), formula_highlight))
        self.play(FadeIn(formula_highlight))
        self.wait(2)
