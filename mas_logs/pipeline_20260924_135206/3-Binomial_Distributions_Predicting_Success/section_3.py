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
            "The Binomial formula calculates success probability.",
            "It uses combinations to arrange successes in n trials.",
            "P(X=k) accounts for successes and failures.",
            "n is the number of trials.",
            "k is the number of successes."
        ]
        self.setup_layout("The Binomial Formula", lecture_lines)
        
        formula = MathTex(
            "P(X=k)", "=", "{n \\choose k}", "\\cdot", "p^k", "\\cdot", "(1-p)^{n-k}",
            font_size=36
        )
        # Apply layout fix as per issue 38 (which covers issues 27, 28, 29)
        self.place_in_area(formula, 'B2', 'D3', scale_factor=0.85)
        
        # Load assets
        asset_icon = None
        try:
            # Asset is just a dummy/placeholder as per path, but must load something if possible or handle gracefully
            asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        except:
            # Fallback if file doesn't exist
            asset_icon = Dot(radius=0.1, color=WHITE)
            
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(formula))
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF69B4"))
        combination = formula[2]
        surround = SurroundingRectangle(combination, color="#FF69B4", buff=0.1)
        self.play(Create(surround))
        self.wait(1)
        self.play(FadeOut(surround))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#7B68EE"))
        prob_part = VGroup(formula[4], formula[5], formula[6])
        surround_prob = SurroundingRectangle(prob_part, color="#7B68EE", buff=0.1)
        self.play(Create(surround_prob))
        self.wait(1)
        self.play(FadeOut(surround_prob))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        # n references: formula[2] (comb) and formula[6] (pow)
        n_mobjects = VGroup(formula[2][0], formula[6][2])
        asset_n = asset_icon.copy().scale(0.5).next_to(n_mobjects, UP)
        self.play(FadeIn(asset_n), Indicate(n_mobjects), asset_n.animate.scale(1.2).set_opacity(0.5).set_opacity(1.0))
        self.wait(1)
        self.play(FadeOut(asset_n))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        k_mobjects = VGroup(formula[0][3], formula[2][2], formula[4][1], formula[6][5])
        asset_k = asset_icon.copy().scale(0.5).next_to(k_mobjects, UP)
        self.play(FadeIn(asset_k), Indicate(k_mobjects), asset_k.animate.scale(1.2).set_opacity(0.5).set_opacity(1.0))
        self.wait(1)
        self.play(FadeOut(asset_k))
