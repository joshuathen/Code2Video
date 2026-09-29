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
        config.no_latex_cleanup = True
        self.setup_layout("Introducing the Taylor Series", ["Complex functions can be infinite sums.", "Taylor series use simple powers as building blocks.", "Stacking these blocks creates complex shapes."])
        
        # Assets
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg"
        
        # === Animation for Lecture Line 1 ===
        # Display the power terms: 1, x, x^2/2!, x^3/3! using blocks.svg as the building blocks for each term. Color in #00FFFF.
        self.lecture[0].set_color("#00FFFF")
        
        terms = ["1", "x", r"\frac{x^2}{2!}", r"\frac{x^3}{3!}"]
        term_mobjects = VGroup(*[MathTex(t) for t in terms])
        
        blocks = VGroup(*[SVGMobject(asset_path).set_color("#00FFFF") for _ in range(4)])
        
        for i, (block, term) in enumerate(zip(blocks, term_mobjects)):
            pos = [f"B{1+i}", f"B{2+i}", f"C{1+i}", f"C{2+i}"][i] # Simplified positioning
            self.place_at_grid(block, f"B{1+i}", scale_factor=0.3)
            term.next_to(block, DOWN, buff=0.1)
            self.play(FadeIn(block), Write(term))

        # === Animation for Lecture Line 2 ===
        # Show how these terms 'stack' linearly to form a path, using blocks.svg to represent the physical stacking process. Animate them appearing one by one.
        self.lecture[1].set_color("#00FFFF")
        
        stack_vgroup = VGroup()
        for i in range(4):
            b = SVGMobject(asset_path).set_color("#00FFFF")
            self.place_at_grid(b, f"D{1+i}", scale_factor=0.3)
            stack_vgroup.add(b)
        
        self.play(LaggedStart(*[FadeIn(b) for b in stack_vgroup], lag_ratio=0.5))

        # === Animation for Lecture Line 3 ===
        # Transform these simple terms into the exponential series curve. Use #FFA500 for the final shape, transitioning from the blocks.svg structure to a smooth line.
        self.lecture[2].set_color("#FFA500")
        
        curve = FunctionGraph(lambda x: np.exp(x/2) - 1, x_range=[-1, 2], color="#FFA500")
        self.place_in_area(curve, "E1", "F6", scale_factor=0.5)
        
        self.play(
            ReplacementTransform(stack_vgroup, curve),
            run_time=2
        )
        self.wait(1)
