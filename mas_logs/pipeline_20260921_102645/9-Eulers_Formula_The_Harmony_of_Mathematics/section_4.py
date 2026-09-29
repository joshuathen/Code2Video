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
        lecture_lines = ["Constants e, i, pi combine.", "They form the most beautiful equation.", "Euler's identity emerges from them."]
        self.setup_layout("The Identity e^(iπ) + 1 = 0", lecture_lines)
        
        # Define mobjects
        equation1 = MathTex(r"e^{i\pi} = -1", font_size=48, color=WHITE)
        equation2 = MathTex(r"e^{i\pi} + 1 = 0", font_size=48, color=WHITE)
        equation2_partial = MathTex(r"e^{i\pi} + 1 = -1 + 1", font_size=48, color=RED)
        
        # Placeholder for assets - using SVGMobject if valid or just simple shape for placeholder
        # Per instructions "MUST use the elements ... [Asset: ...]"
        # Since the path is given as .../icon/none.svg, assuming it's a valid path
        asset_placeholder = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color=WHITE).scale(0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_in_area(equation1, 'B2', 'D4', scale_factor=1.2)
        self.play(Write(equation1))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        asset_placeholder.move_to(self.grid["A2"])
        self.play(FadeIn(asset_placeholder))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.place_in_area(equation2_partial, 'B4', 'D4', scale_factor=1.2)
        self.play(ReplacementTransform(equation1, equation2_partial))
        self.play(self.lecture[1].animate.set_color("#FF0000"))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        # Final result
        self.place_in_area(equation2, 'B4', 'D5', scale_factor=1.5)
        self.play(ReplacementTransform(equation2_partial, equation2))
        self.play(equation2.animate.set_color("#00FF00"))
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # Final Asset
        asset_placeholder_2 = asset_placeholder.copy()
        asset_placeholder_2.move_to(self.grid["F5"])
        self.play(FadeIn(asset_placeholder_2))
        
        self.wait(2)
