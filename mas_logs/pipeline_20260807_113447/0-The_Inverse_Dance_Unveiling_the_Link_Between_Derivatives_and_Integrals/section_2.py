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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The Big Question: Are They Related?", 
            [
                "Are these two concepts actually connected?", 
                "Think of them as inverse operations.", 
                "Like multiplication reversing a division."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        ddx = MathTex(r"\frac{d}{dx}", color=BLUE_A)
        q_mark = Text("?", color="#FFFF00", font_size=48)
        integral = MathTex(r"\int", color=PURPLE_A)
        
        # Issue 32: Correcting scale factor for ddx and integral symbols to 1.0
        self.place_at_grid(ddx, "B3", scale_factor=1.0)
        self.place_at_grid(q_mark, "B4", scale_factor=1.0)
        self.place_at_grid(integral, "B5", scale_factor=1.0)
        
        self.play(FadeIn(ddx), FadeIn(q_mark), FadeIn(integral))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(GREEN_A),
            FadeOut(ddx), FadeOut(q_mark), FadeOut(integral)
        )
        
        # Issue 26: Integrating Magic Math Machine asset
        machine_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg").set_color(GREEN_C)
        self.place_in_area(machine_asset, "C3", "E5", scale_factor=2.5)
        
        # Issue 33: Placing machine label in area C3-C5
        machine_label = Text("Magic Math Machine", font_size=18, color=GREEN_B)
        self.place_in_area(machine_label, "C3", "C5", scale_factor=0.7)
        
        # Issue 26: Integrating Lever assets
        deriv_lever_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lever.svg").set_color(GREEN_A)
        deriv_lever_label = Text("Derivative", font_size=14, color=GREEN_A)
        deriv_lever_group = VGroup(deriv_lever_svg, deriv_lever_label.next_to(deriv_lever_svg, DOWN, buff=0.1))
        
        integ_lever_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lever.svg").set_color(GREEN_A)
        integ_lever_label = Text("Integral", font_size=14, color=GREEN_A)
        integ_lever_group = VGroup(integ_lever_svg, integ_lever_label.next_to(integ_lever_svg, DOWN, buff=0.1))

        # Issue 34: Scaling lever groups to 0.8 for better margins
        self.place_at_grid(deriv_lever_group, "C6", scale_factor=0.8)
        self.place_at_grid(integ_lever_group, "E6", scale_factor=0.8)
        
        self.play(DrawBorderThenFill(machine_asset), Write(machine_label), FadeIn(deriv_lever_group), FadeIn(integ_lever_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(ORANGE)
        )
        
        # Function flow demonstration
        f_x2 = MathTex("x^2", color=ORANGE)
        f_2x = MathTex("2x", color=ORANGE)
        f_x2_return = MathTex("x^2", color=ORANGE)
        
        self.place_at_grid(f_x2, "A4", scale_factor=1.2)
        
        # Step 1: x^2 enters machine
        self.play(FadeIn(f_x2))
        self.play(f_x2.animate.move_to(self.grid["D4"]))
        
        # Step 2: Derivative lever pull and transformation
        self.play(deriv_lever_svg.animate.rotate(-PI/4, about_point=deriv_lever_svg.get_bottom()), run_time=0.4)
        self.play(ReplacementTransform(f_x2, f_2x.move_to(self.grid["D4"])))
        self.play(deriv_lever_svg.animate.rotate(PI/4, about_point=deriv_lever_svg.get_bottom()), run_time=0.4)
        
        # Step 3: 2x moves out to exit point
        self.play(f_2x.animate.move_to(self.grid["F4"]))
        self.wait(0.5)
        
        # Step 4: 2x moves back in for reversal
        self.play(f_2x.animate.move_to(self.grid["D4"]))
        
        # Step 5: Integral lever pull and back-transformation
        self.play(integ_lever_svg.animate.rotate(-PI/4, about_point=integ_lever_svg.get_bottom()), run_time=0.4)
        self.play(ReplacementTransform(f_2x, f_x2_return.move_to(self.grid["D4"])))
        self.play(integ_lever_svg.animate.rotate(PI/4, about_point=integ_lever_svg.get_bottom()), run_time=0.4)
        
        # Step 6: Final x^2 returns to top
        self.play(f_x2_return.animate.move_to(self.grid["A4"]))
        self.wait(2)
