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
        # Title and lecture lines
        title_text = "Finding the Shortcut (The 'Why')"
        lecture_lines = [
            "- Instead of two steps, we want one \"Master Matrix\".",
            "- The product matrix C performs both transformations at once.",
            "- Multiplication order is right-to-left: C = BA."
        ]
        self.setup_layout(title_text, lecture_lines)

        cat_asset_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cat.png"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(ORANGE)
        
        # Display the original 'Cyber-Cat' next to the final transformed cat for comparison.
        cat_orig = ImageMobject(cat_asset_path)
        self.place_at_grid(cat_orig, "B2", scale_factor=0.8)
        label_orig = Text("Original", font_size=16, color=WHITE).next_to(cat_orig, DOWN, buff=0.1)
        
        cat_final = ImageMobject(cat_asset_path)
        cat_final.rotate(90 * DEGREES).stretch(1.5, dim=0) 
        self.place_at_grid(cat_final, "B4", scale_factor=0.8) # Issue 29: Positioned at B4
        label_final = Text("Result", font_size=16, color=WHITE).next_to(cat_final, DOWN, buff=0.1)

        # Introduce 'Master Matrix' C = BA in #FFA500 as a single shortcut.
        label_c_header = Text("Master Matrix C", color="#FFA500", font_size=24)
        self.place_at_grid(label_c_header, "A3")

        self.play(FadeIn(cat_orig), Write(label_orig))
        self.play(FadeIn(cat_final), Write(label_final))
        self.play(Write(label_c_header))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(ORANGE))
        
        # Create a 'Teleportation Gate' visual as a glowing #00FFFF rectangle.
        gate = Rectangle(width=1.2, height=1.5, color="#00FFFF", stroke_width=4)
        gate.set_fill("#00FFFF", opacity=0.3)
        self.place_at_grid(gate, "B3")
        gate_glow = gate.copy().set_stroke(width=8, opacity=0.5)
        
        self.play(Create(gate), Create(gate_glow))
        
        # Animate the cat entering the gate and instantly exiting in its final form.
        moving_cat = ImageMobject(cat_asset_path)
        self.place_at_grid(moving_cat, "B2", scale_factor=0.8)
        
        # Entering Gate
        self.play(moving_cat.animate.move_to(self.grid["B3"]), run_time=1.2)
        
        # Transformation Flash
        flash = Flash(self.grid["B3"], color="#00FFFF", line_length=0.3)
        
        # Exiting Gate in transformed form
        transformed_copy = ImageMobject(cat_asset_path).rotate(90 * DEGREES).stretch(1.5, dim=0)
        self.place_at_grid(transformed_copy, "B3", scale_factor=0.8)
        
        self.play(
            FadeOut(moving_cat),
            FadeIn(transformed_copy),
            flash,
            run_time=0.4
        )
        
        self.play(transformed_copy.animate.move_to(self.grid["B4"]), run_time=1.2)
        self.play(FadeOut(transformed_copy))
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(ORANGE))
        
        # Formula C = BA [Issue 30: Place at D3, scale 0.9]
        formula_c = Text("C=BA", font_size=48)
        self.place_at_grid(formula_c, "D3", scale_factor=0.9)
        
        self.play(Write(formula_c))
        
        # Highlight the right-to-left order of B then A in the expression BA using #FF00FF.
        h_a = SurroundingRectangle(formula_c[3], color="#FF00FF", buff=0.1)
        h_b = SurroundingRectangle(formula_c[2], color="#FF00FF", buff=0.1)
        
        label_a = Text("1st (A)", color="#FF00FF", font_size=16).next_to(h_a, DOWN, buff=0.1)
        label_b = Text("2nd (B)", color="#FF00FF", font_size=16).next_to(h_b, DOWN, buff=0.1)
        
        self.play(Create(h_a), Write(label_a))
        self.wait(0.5)
        self.play(Create(h_b), Write(label_b))
        
        self.wait(2)
