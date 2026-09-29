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
        self.setup_layout("The Wallis Product Formula", [
            "The Wallis Product defines pi/2 as a product.",
            "Alternating fractions build the target value.",
            "Each ingredient improves the flavor meter's precision.",
            "Convergence shows pi emerging from simple ratios.",
            "The infinite product mirrors the circle's nature."
        ])
        
        # === Animation for Lecture Line 1 ===
        wallis_formula = MathTex(
            r"\frac{\pi}{2} = \prod_{n=1}^{\infty} \left( \frac{2n}{2n-1} \cdot \frac{2n}{2n+1} \right)",
            color="#57FFFF"
        )
        # Using SVG asset
        gauge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/meter.svg")
        self.place_at_grid(gauge, "A4", scale_factor=0.5)
        
        # Applying layout fix for Issue 30/39
        self.place_in_area(wallis_formula, "B3", "B6", scale_factor=0.8)
        self.play(Write(wallis_formula), FadeIn(gauge))
        self.lecture[0].set_color("#57FFFF")
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        factors = VGroup(
            MathTex(r"\left( \frac{2}{1} \cdot \frac{2}{3} \right)"),
            MathTex(r"\left( \frac{4}{3} \cdot \frac{4}{5} \right)"),
            MathTex(r"\left( \frac{6}{5} \cdot \frac{6}{7} \right)")
        ).arrange(RIGHT, buff=0.2).scale(0.8)
        
        # Applying layout fix for Issue 31/39
        self.place_in_area(factors, "C3", "C6", scale_factor=0.9)
        
        for factor in factors:
            self.play(FadeIn(factor))
        self.lecture[1].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        flavor_meter = Rectangle(width=4, height=0.5, color=WHITE).next_to(factors, DOWN, buff=1)
        # flavor_meter is moved/placed using grid, keeping logic consistent
        self.place_at_grid(flavor_meter, "E3")
        label = Text("Flavor Meter", font_size=20).next_to(flavor_meter, UP)
        self.add(flavor_meter, label)
        self.lecture[2].set_color("#00FF00")
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF57FF")
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        final_formula = MathTex(r"\frac{\pi}{2}", color="#FFFFFF")
        
        # Applying layout fix for Issue 32/39
        self.place_at_grid(final_formula, "E4", scale_factor=1.2)
        self.play(FadeOut(factors), FadeOut(flavor_meter), FadeOut(label), FadeOut(gauge), ReplacementTransform(wallis_formula, final_formula))
        self.lecture[4].set_color("#FFFFFF")
        self.wait(2)
