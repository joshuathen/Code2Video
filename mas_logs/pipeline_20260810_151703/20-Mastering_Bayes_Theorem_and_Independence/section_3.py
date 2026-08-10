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
        lecture_lines = [
            "Bayes' Theorem updates beliefs with new evidence.",
            "Think like a sniffer dog seeking truth.",
            "Start with a prior belief.",
            "Adjust it based on new likelihood data.",
            "Arrive at a refined posterior probability."
        ]
        self.setup_layout("Introducing Bayes' Theorem", lecture_lines)
        
        # Assets
        dog_icon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/dog.svg"
        
        # Elements
        prior_dog = SVGMobject(dog_icon_path, color=WHITE)
        prior_text = Text("Prior", font_size=18).next_to(prior_dog, DOWN)
        prior_group = VGroup(prior_dog, prior_text)
        
        evidence_node = Circle(radius=0.4, color="#FFFF00", fill_opacity=0.5)
        evidence_text = Text("Evidence", font_size=18).move_to(evidence_node)
        evidence_group = VGroup(evidence_node, evidence_text)
        
        formula = MathTex(r"P(H|E) = \frac{P(E|H)P(H)}{P(E)}", color=WHITE, font_size=36)
        
        posterior_node = Circle(radius=0.5, color="#00FFFF", fill_opacity=0.7)
        posterior_text = Text("Posterior", font_size=18).move_to(posterior_node)
        posterior_group = VGroup(posterior_node, posterior_text)
        
        posterior_dog = SVGMobject(dog_icon_path, color=WHITE).scale(0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFCC00"))
        self.place_at_grid(prior_group, "B3", scale_factor=0.7)
        self.play(FadeIn(prior_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.place_at_grid(evidence_group, "B4", scale_factor=0.7)
        self.play(Create(evidence_group))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.place_in_area(formula, "C2", "C5", scale_factor=0.6)
        self.play(Write(formula))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF0000"))
        self.play(DrawBorderThenFill(posterior_node))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF0000"))
        self.place_at_grid(posterior_group, "E4", scale_factor=0.8)
        self.play(FadeIn(posterior_group), FadeIn(posterior_dog.next_to(posterior_group, DOWN)))
        self.wait(1)
