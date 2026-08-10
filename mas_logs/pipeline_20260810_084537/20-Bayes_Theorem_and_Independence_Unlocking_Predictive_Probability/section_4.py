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
        self.setup_layout("Synthesis & Application: The Diagnostic Power", [
            "Independence simplifies our Bayesian probability calculations.", 
            "Dependent events use the likelihood ratio to update.", 
            "Bayes' prevents panic by checking prior probabilities."
        ])
        
        # Elements
        test_diagram = VGroup(
            Circle(radius=0.5, color=WHITE),
            Text("Test", font_size=20)
        )
        pos_result_a = Dot(color=RED, radius=0.15)
        pos_result_b = Dot(color=RED, radius=0.15)
        posterior_eqn = MathTex(r"P(H|E) = \frac{P(E|H)P(H)}{P(E)}", color=GREEN)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_at_grid(test_diagram, 'B2', scale_factor=1.0)
        self.play(FadeIn(test_diagram))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        self.place_at_grid(pos_result_a, 'C4')
        self.place_at_grid(pos_result_b, 'D4')
        self.play(FadeIn(pos_result_a), FadeIn(pos_result_b))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        self.place_in_area(posterior_eqn, 'E1', 'F6', scale_factor=0.8)
        self.play(Write(posterior_eqn))
        self.wait(2)
