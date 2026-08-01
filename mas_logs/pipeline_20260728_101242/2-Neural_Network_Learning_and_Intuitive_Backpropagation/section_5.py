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

class Section5Scene(TeachingScene):
    def construct(self):
        title_text = "The Chain Rule: A Ripple Effect"
        lecture_lines = [
            "The chain rule calculates the ripple of error.",
            "It measures weight sensitivity through the network.",
            "A small nudge at the start affects everything.",
            "Gradients tell us the direction of steepest slope.",
            "We follow these slopes to reduce overall error."
        ]
        self.setup_layout(title_text, lecture_lines)

        # --- Object Creation ---
        # White dominos [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/domino.svg]
        dominos = VGroup(*[
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/domino.svg").set_color(WHITE)
            for _ in range(3)
        ])
        for i, d in enumerate(dominos):
            self.place_at_grid(d, f"C{i+1}", scale_factor=0.6)
        
        # Error box
        error_box = Rectangle(height=0.8, width=1.2, color="#FF4500", fill_opacity=1)
        error_text = Text("Error", font_size=18, color=WHITE)
        error_group = VGroup(error_box, error_text)
        
        # Issue 40: Fix positioning - anchoring in area C2-C6 (centers at C4)
        self.place_in_area(error_group, 'C2', 'C6', scale_factor=0.8)

        # Nudge arrow (Yellow)
        nudge_arrow = Arrow(start=LEFT, end=RIGHT, color="#FFD700", stroke_width=4).scale(0.4)
        nudge_arrow.next_to(dominos[0], LEFT, buff=0.3)

        # Sensitivity labels (Cyan)
        sens_labels = VGroup(*[
            MathTex(r"\frac{\partial E}{\partial w}", font_size=20, color="#00FFFF")
            for _ in range(3)
        ])
        for i, l in enumerate(sens_labels):
            self.place_at_grid(l, f"B{i+1}", scale_factor=1.0)

        # Gradient arrows (Green)
        grad_arrows = VGroup(*[
            Arrow(start=DOWN, end=UP, color="#00FF00", stroke_width=4).scale(0.3)
            for _ in range(3)
        ])
        for i, a in enumerate(grad_arrows):
            a.next_to(dominos[i], UP, buff=0.2)

        # Chain Rule Label (Green) - Issue 41: Fix positioning at D4
        chain_rule_label = Text("Chain Rule", color="#00FF00", font_size=24)
        self.place_at_grid(chain_rule_label, 'D4', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # "The chain rule calculates the ripple of error."
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(FadeIn(dominos), FadeIn(error_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "It measures weight sensitivity through the network."
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.play(FadeIn(nudge_arrow))
        # Nudge the first domino
        self.play(dominos[0].animate.rotate(-PI/4, about_point=dominos[0].get_bottom()), run_time=0.6)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "A small nudge at the start affects everything."
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        # Ripple effect through dominos
        ripple_anims = [
            dominos[i].animate.rotate(-PI/4, about_point=dominos[i].get_bottom())
            for i in range(1, 3)
        ]
        self.play(Succession(*ripple_anims, lag_ratio=0.5))
        self.play(FadeIn(sens_labels))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Gradients tell us the direction of steepest slope."
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        self.play(Create(grad_arrows))
        self.play(Write(chain_rule_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "We follow these slopes to reduce overall error."
        self.play(self.lecture[4].animate.set_color("#FF4500"))
        # Error box shifts position significantly
        self.play(
            error_group.animate.shift(RIGHT * 1.5),
            error_group.animate.set_fill(opacity=0.5),
            run_time=1.5
        )
        self.wait(2)
