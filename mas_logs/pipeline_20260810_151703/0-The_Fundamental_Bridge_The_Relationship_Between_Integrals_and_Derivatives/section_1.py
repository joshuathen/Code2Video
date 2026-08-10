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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Calculus explores two core concepts: derivatives and integrals.",
            "Derivatives analyze the slope of a curve locally.",
            "Integrals calculate the total area accumulated under curves."
        ]
        self.setup_layout("Prerequisite Warm-up: The Tangent vs. The Area", lecture_lines)
        
        # Elements
        tangent_label = MathTex(r"\\frac{dy}{dx}", color="#FFD700")
        area_label = MathTex(r"\\int f(x) dx", color="#00BFFF")
        formula_group = VGroup(tangent_label, area_label)
        
        # Load asset
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_at_grid(tangent_label, "B3", scale_factor=1.2)
        self.place_at_grid(area_label, "D3", scale_factor=1.2)
        self.place_at_grid(asset_icon, "C5", scale_factor=0.5)
        self.play(FadeIn(tangent_label), FadeIn(area_label), FadeIn(asset_icon))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(GRAY), FadeIn(self.lecture[1]))
        tangent_line = Line(start=LEFT*0.5, end=RIGHT*0.5, color="#FFD700").rotate(PI/4).move_to(self.grid["B3"])
        self.play(Create(tangent_line))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(GRAY), FadeIn(self.lecture[2]))
        area_rect = Rectangle(width=0.8, height=0.8, color="#00BFFF", fill_opacity=0.3).move_to(self.grid["D3"])
        self.play(Create(area_rect))
        
        # Highlight relationship
        self.play(
            tangent_label.animate.set_color(WHITE),
            area_label.animate.set_color(WHITE),
            run_time=1
        )
        self.wait(1)
