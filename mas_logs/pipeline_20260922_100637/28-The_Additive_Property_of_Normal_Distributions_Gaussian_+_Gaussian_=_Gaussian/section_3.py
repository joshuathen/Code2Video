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
        self.setup_layout("The Core Theorem: Adding the 'DNA'", [
            "Adding independent Gaussians is surprisingly elegant.",
            "The resulting mean is the sum of means.",
            "The resulting variance is the sum of variances.",
            "The two bell curves merge into one.",
            "The final shape is always another Gaussian."
        ])
        
        # Axes
        axes = Axes(x_range=[-4, 8, 1], y_range=[0, 1.2, 0.5], axis_config={"include_numbers": False}).scale(0.6)
        self.place_in_area(axes, "B2", "E5")
        self.add(axes)

        # Asset
        dna_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dna.svg")
        self.place_at_grid(dna_icon, "A6", scale_factor=0.3)

        # Curves
        curve1 = axes.plot(lambda x: 0.8 * np.exp(-(x - 0)**2 / (2 * 0.5**2)), color=WHITE)
        curve2 = axes.plot(lambda x: 0.8 * np.exp(-(x - 2)**2 / (2 * 0.8**2)), color=WHITE)
        combined_curve = axes.plot(lambda x: 0.6 * np.exp(-(x - 2)**2 / (2 * 1.3**2)), color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Create(curve1), Create(curve2), FadeIn(dna_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFCC00"))
        mean_text = MathTex(r"\mu_{total} = \mu_1 + \mu_2", color="#FFCC00")
        self.place_at_grid(mean_text, "A2")
        self.play(Write(mean_text))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFCC00"))
        var_text = MathTex(r"\sigma^2_{total} = \sigma^2_1 + \sigma^2_2", color="#FFCC00")
        self.place_at_grid(var_text, "A3")
        self.play(Write(var_text))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        self.play(ReplacementTransform(VGroup(curve1, curve2), combined_curve))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        final_label = MathTex(r"N(\mu_1+\mu_2, \sigma^2_1+\sigma^2_2)", color="#00FF00")
        self.place_at_grid(final_label, "F4", scale_factor=0.8)
        
        dna_icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dna.svg")
        self.place_at_grid(dna_icon2, "F6", scale_factor=0.3)
        
        self.play(Write(final_label), FadeIn(dna_icon2))
        self.wait(2)
