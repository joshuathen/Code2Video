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
            "Multiply Queries by Keys to get scores.",
            "Divide by root-d_k then apply Softmax.",
            "Multiply weights by Values for output."
        ]
        self.setup_layout("Visualizing Scaled Dot-Product", lecture_lines)
        
        # Assets
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        thermometer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thermometer.svg")
        scales_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scales.svg")
        
        # Elements
        q_label = Text("Q", color=BLUE).scale(0.8)
        k_label = Text("K", color=YELLOW).scale(0.8)
        dot_product = MathTex(r"Q \cdot K^T", color=WHITE)
        v_label = Text("V", color=RED).scale(0.8)
        softmax_label = Text("Softmax", color=GREEN).scale(0.8)
        
        # Initial positions
        self.place_at_grid(q_label, "B2")
        self.place_at_grid(k_label, "B4")
        self.place_at_grid(v_label, "E3")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(calc_icon, "B3", scale_factor=0.3)
        self.play(FadeIn(q_label), FadeIn(k_label), FadeIn(calc_icon))
        
        dot = Dot(color=WHITE)
        self.place_at_grid(dot, "B3", scale_factor=0.6)
        self.play(Indicate(q_label), Indicate(k_label))
        
        self.place_at_grid(thermometer_icon, "B5", scale_factor=0.3)
        thermometer_icon.set_color("#FF00FF")
        self.play(Transform(dot, dot_product.move_to(self.grid["B3"])), FadeIn(thermometer_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.place_at_grid(softmax_label, "C3", scale_factor=0.7)
        self.place_at_grid(scales_icon, "C5", scale_factor=0.3)
        self.play(FadeIn(softmax_label), FadeIn(scales_icon))
        self.play(dot.animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.play(FadeIn(v_label))
        self.play(Indicate(softmax_label), Indicate(v_label))
        result = Text("Output", color=WHITE).scale(0.7)
        self.place_at_grid(result, "E4", scale_factor=0.8)
        self.play(FadeIn(result))
        self.wait(2)
