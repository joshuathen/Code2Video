from manim import *
import numpy as np

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
    """
    Implementation of Section 4: The Core Reveal: The CLT Definition.
    Visualizes the transformation of population distributions into a Normal distribution
    as sample size increases.
    """
    def construct(self):
        # 1. Setup Layout
        self.setup_layout(
            "The Core Reveal: The CLT Definition",
            [
                'This is the magic of the Central Limit Theorem.', 
                'Any population shape eventually becomes a normal distribution.', 
                'As sample size grows, the curve smooths out.', 
                'The mean of means equals the population mean.', 
                'The spread shrinks by the square root of n.'
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFEE58")
        clt_title = Text("Central Limit Theorem", font_size=36, color="#FFEE58")
        clt_glow = clt_title.copy().set_stroke(width=10, color="#FFEE58", opacity=0.3)
        clt_group = VGroup(clt_glow, clt_title)
        # Resolved Issue 42: Adjusted scale_factor to 0.9 for better visual hierarchy
        self.place_in_area(clt_group, "A2", "A5", scale_factor=0.9)
        
        self.play(Write(clt_title), FadeIn(clt_glow))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#66BB6A")
        
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 1.2, 0.5],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": False}
        )
        # Resolved Issue 43: Reduced scale to 0.8 to prevent overlap with formula
        self.place_in_area(axes, "C1", "F6", scale_factor=0.8)
        self.play(Create(axes))

        # Define different distribution shapes
        def uniform_dist(x):
            return 0.4 if -2 <= x <= 2 else 0
        
        def bimodal_dist(x):
            return 0.6 * (np.exp(-(x+1.5)**2 / 0.5) + np.exp(-(x-1.5)**2 / 0.5))
            
        def normal_dist(x, sigma=1.0):
            return np.exp(-x**2 / (2 * sigma**2))

        dist_uniform = axes.plot(uniform_dist, color=BLUE, use_smoothing=False)
        dist_bimodal = axes.plot(bimodal_dist, color=ORANGE)
        dist_normal = axes.plot(lambda x: normal_dist(x, 0.8), color="#66BB6A")

        self.play(Create(dist_uniform))
        self.wait(0.5)
        self.play(ReplacementTransform(dist_uniform, dist_bimodal))
        self.wait(0.5)
        self.play(ReplacementTransform(dist_bimodal, dist_normal))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#42A5F5")
        # Shrink sigma to show smoothing and narrowing
        dist_narrow = axes.plot(lambda x: normal_dist(x, 0.4), color="#42A5F5")
        self.play(Transform(dist_normal, dist_narrow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF7043")
        mean_line = axes.get_vertical_line(axes.c2p(0, normal_dist(0, 0.4)), color="#FF7043")
        # Fixed: Changed MathTex to Text with Unicode to avoid LaTeX FileNotFoundError
        mean_label = Text("μ_x = μ", font_size=24, color="#FF7043").next_to(mean_line, UP)
        self.play(Create(mean_line), Write(mean_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#AB47BC")
        # Fixed: Changed MathTex to Text with Unicode to avoid LaTeX FileNotFoundError
        std_err_formula = Text("σ_x = σ / √n", font_size=30, color="#AB47BC")
        # Resolved Issue 41: Reduced area and scale to 0.8 to avoid congestion
        self.place_in_area(std_err_formula, "B2", "B5", scale_factor=0.8)
        self.play(Write(std_err_formula))
        
        # Visualize shrinking spread
        dist_very_narrow = axes.plot(lambda x: normal_dist(x, 0.2), color="#AB47BC")
        self.play(Transform(dist_normal, dist_very_narrow))
        self.wait(2)
