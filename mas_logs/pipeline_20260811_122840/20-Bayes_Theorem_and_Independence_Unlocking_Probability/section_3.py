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
        lecture_lines = ["Bayes' Theorem lets us update our beliefs.", "We combine a prior with new evidence.", "This gives us a refined posterior probability.", "It is essential for diagnostic testing.", "Evidence shifts the probability weight effectively."]
        self.setup_layout("Introduction to Bayes' Theorem", lecture_lines)
        
        # Asset images
        asset_test = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/test.svg")
        asset_patient = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/patient.svg")
        
        # === Animation for Lecture Line 1 ===
        # P(A|B) = [P(B|A) * P(A)] / P(B)
        formula = MathTex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}", color=WHITE)
        self.place_in_area(formula, 'B1', 'C3', scale_factor=0.7)
        self.place_at_grid(asset_test, 'B5', scale_factor=0.5)
        self.play(Write(formula), FadeIn(asset_test))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Prior P(A) as a bar chart. Color #888888.
        prior_bar = Rectangle(width=0.8, height=2.0, color="#888888", fill_opacity=0.8)
        prior_label = Text("Prior P(A)", font_size=18).next_to(prior_bar, DOWN)
        prior_group = VGroup(prior_bar, prior_label)
        self.place_at_grid(prior_group, 'D2', scale_factor=0.6)
        self.play(Create(prior_group))
        self.lecture[1].set_color(BLUE)

        # === Animation for Lecture Line 3 ===
        # Introduce Evidence B as a light pulse moving toward Prior bar. Color #FFCC00.
        evidence_pulse = Circle(radius=0.3, color="#FFCC00", fill_opacity=0.6)
        self.place_at_grid(evidence_pulse, 'B6', scale_factor=1.0)
        self.play(evidence_pulse.animate.move_to(prior_bar.get_center()), run_time=1.5)
        self.lecture[2].set_color(BLUE)

        # === Animation for Lecture Line 4 ===
        # Animate Prior bar changing size to Posterior P(A|B). Color #FF5555.
        posterior_bar = Rectangle(width=0.8, height=0.5, color="#FF5555", fill_opacity=0.8)
        posterior_bar.move_to(prior_bar.get_center())
        self.play(ReplacementTransform(prior_bar, posterior_bar), run_time=1.5)
        posterior_label = Text("Posterior P(A|B)", font_size=18).next_to(posterior_bar, DOWN)
        self.play(Write(posterior_label))
        self.lecture[3].set_color(BLUE)

        # === Animation for Lecture Line 5 ===
        # Add a summary label 'Updated Belief'. Color #FFFFFF.
        summary = Text("Updated Belief", color=WHITE, font_size=24)
        self.place_at_grid(summary, 'E5', scale_factor=0.8)
        self.place_at_grid(asset_patient, 'E6', scale_factor=0.5)
        self.play(FadeIn(summary), FadeIn(asset_patient))
        self.lecture[4].set_color(BLUE)
        self.wait(1)
