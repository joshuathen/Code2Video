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
        self.setup_layout("The Math: Dot-Product Attention", [
            "Scores measure similarity between Queries and Keys.",
            "Higher dot-products indicate stronger relevance scores.",
            "Softmax converts scores into percentage weightings."
        ])
        
        # Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        formula = MathTex(
            r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            font_size=36
        )
        # Applying fix for issue 30
        self.place_in_area(formula, 'B1', 'D6', scale_factor=0.6)
        
        # Applying fix for issue 31
        attention_label = Text("Attention", font_size=24)
        self.place_at_grid(attention_label, 'D4', scale_factor=0.8)
        
        # Define parts for color changes
        formula.set_color(WHITE)
        qk_t = formula.get_part_by_tex("QK^T")
        sqrt_dk = formula.get_part_by_tex(r"\sqrt{d_k}")
        softmax = formula.get_part_by_tex(r"\text{softmax}")
        v = formula[-1]

        # Asset implementation (issue 19)
        # Using a dummy rectangle since the asset path is /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # As per instructions, prefer SVGMobject if available or Tex/shapes.
        # Given "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", I will create a placeholder icon
        asset_icon = Square(color=WHITE, fill_opacity=1, side_length=0.5)
        # Applying fix for issue 32
        self.place_in_area(asset_icon, 'A4', 'F6', scale_factor=0.75)
        
        # === Animation for Lecture Line 1 ===
        self.play(Write(formula), Write(attention_label))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            qk_t.animate.set_color("#00CED1"),
            sqrt_dk.animate.set_color("#FFD700")
        )
        self.lecture[1].set_color("#00CED1")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            softmax.animate.set_color("#FF4500"),
            v.animate.set_color("#32CD32"),
            FadeIn(asset_icon)
        )
        self.lecture[2].set_color("#FF4500")
        self.wait(2)
