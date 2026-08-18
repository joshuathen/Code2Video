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
            "Independence simplifies Bayes' complex calculations.",
            "When independent, the diagnostic power vanishes.",
            "The formula reduces to the prior probability."
        ]
        self.setup_layout("Synthesis: Independence in Bayes", lecture_lines)
        
        # Elements for animation
        dependent_text = Text("Dependent (Diagnostic)", font_size=24, color=YELLOW)
        independent_text = Text("Independent (No Update)", font_size=24, color=BLUE)
        
        # Load asset
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        # Dependent visuals
        prior_node = Circle(radius=0.3, color=WHITE).set_fill(GREY, 0.5)
        sensor_node = Circle(radius=0.3, color=WHITE).set_fill(GREY, 0.5)
        influence_arrow = Arrow(prior_node.get_center(), sensor_node.get_center(), buff=0.4, color=YELLOW)
        
        dependent_group = VGroup(prior_node, sensor_node, influence_arrow, asset_icon)
        
        # Applying requested fixes
        self.place_at_grid(dependent_group, "C3", scale_factor=1.0)
        self.place_at_grid(dependent_text, "B3", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(dependent_group), Write(dependent_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        independent_group = dependent_group.copy()
        independent_group.set_color(BLUE)
        influence_arrow.set_color(BLUE)
        
        self.play(
            ReplacementTransform(dependent_group.copy(), independent_group),
            FadeOut(influence_arrow),
            ReplacementTransform(dependent_text, independent_text)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        reduction_tex = MathTex("P(A|B) = P(A)", color=BLUE)
        
        # Applying requested fix
        self.place_in_area(reduction_tex, "D3", "E5", scale_factor=0.9)
        self.play(Write(reduction_tex))
        self.wait(2)
