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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initialize layout
        title_text = "Interactive Summary & Visual Recap"
        lecture_lines = [
            "Product rule adds growth; Chain rule multiplies rates.",
            "Visualize rectangles for products and gears for composition.",
            "Master these patterns to tackle any calculus challenge."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        color1 = YELLOW
        self.play(self.lecture[0].animate.set_color(color1))

        # Side-by-side formulas
        # Issue 35 fix: use B2 instead of A1 (scale_factor=0.8 logic applied)
        product_formula = MathTex(r"\frac{d}{dx}[u \cdot v] = u'v + uv'", color=color1)
        self.place_at_grid(product_formula, "B2", scale_factor=0.6)
        
        chain_formula = MathTex(r"\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)", color=color1)
        self.place_at_grid(chain_formula, "B5", scale_factor=0.6)

        self.play(Write(product_formula), Write(chain_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        color2 = BLUE_B
        self.play(self.lecture[1].animate.set_color(color2))

        # Rectangle icon for Product
        # Issue 36 fix: use D2 instead of C3 to avoid clustering
        rect_icon = Rectangle(width=2, height=1.5, color=BLUE, fill_opacity=0.3)
        self.place_at_grid(rect_icon, "D2", scale_factor=0.6)

        # Gear icon for Chain
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/gears.svg]
        gear_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/gears.svg", color=BLUE_A)
        self.place_at_grid(gear_icon, "D5", scale_factor=0.6)

        self.play(Create(rect_icon), FadeIn(gear_icon))
        self.wait(1)

        # Merge into single 'Calculus Rules' symbol
        # Animation: icons merge into a single 'Calculus Rules' symbol.
        calculus_rules_text = Text("Calculus Rules", color=WHITE, weight=BOLD)
        self.place_in_area(calculus_rules_text, "D3", "D4", scale_factor=0.8)

        self.play(
            ReplacementTransform(VGroup(rect_icon, gear_icon), calculus_rules_text),
            product_formula.animate.scale(0.8).move_to(self.grid["A2"]),
            chain_formula.animate.scale(0.8).move_to(self.grid["A5"])
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        color3 = GREEN
        self.play(self.lecture[2].animate.set_color(color3))

        # Mnemonic: 'Add Growth for Products, Multiply Rates for Chains' appears in #00FF00
        # Issue 34 fix: place at E5 instead of F6
        mnemonic = Text("Add Growth for Products,\nMultiply Rates for Chains", color="#00FF00", font_size=26)
        self.place_at_grid(mnemonic, "E5", scale_factor=0.8)

        self.play(FadeIn(mnemonic, shift=UP))
        self.play(Indicate(mnemonic, color=GREEN_A))
        self.wait(2)
