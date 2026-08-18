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
        self.setup_layout("The Unified Foundation: Prerequisite Review", [
            "Exponents are just repeated multiplication.", 
            "2 cubed means 2 times 2 times 2.", 
            "Imagine a side length of 2 growing to volume 8."
        ])
        self.lecture.set_opacity(1)

        # === Animation for Lecture Line 1 ===
        # Exponents are just repeated multiplication.
        self.lecture[0].set_color(YELLOW)
        eqn = MathTex("b^n = x")
        # Fixing per issue 26/41: Use B2
        self.place_at_grid(eqn, 'B2', scale_factor=1.2)
        
        # Adding Asset
        cube_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        self.place_at_grid(cube_icon, 'B5', scale_factor=0.5)

        self.play(FadeIn(eqn), FadeIn(cube_icon))
        
        # Highlight base b, exponent n, power x
        b_part = eqn.get_part_by_tex("b")
        n_part = eqn.get_part_by_tex("n")
        x_part = eqn.get_part_by_tex("x")
        
        # Use B020: scale_factor 0.7-0.8 for labels
        # Use B011: tethered labels
        base_label = Text("Base", font_size=20, color=YELLOW).scale(0.75)
        base_label.next_to(b_part, DOWN)
        
        exp_label = Text("Exponent", font_size=20, color=BLUE).scale(0.75)
        exp_label.next_to(n_part, UP)
        
        pow_label = Text("Power", font_size=20, color=GREEN).scale(0.75)
        pow_label.next_to(x_part, DOWN)

        self.play(
            b_part.animate.set_color(YELLOW),
            n_part.animate.set_color(BLUE),
            x_part.animate.set_color(GREEN),
            Write(base_label),
            Write(exp_label),
            Write(pow_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # 2 cubed means 2 times 2 times 2.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        calc = MathTex("2^3 = 2 \\times 2 \\times 2 = 8")
        # Fixing per issue 27/42: Use D2
        self.place_at_grid(calc, 'D2', scale_factor=1.0)
        self.play(FadeIn(calc), FadeOut(eqn), FadeOut(base_label), FadeOut(exp_label), FadeOut(pow_label), FadeOut(cube_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Imagine a side length of 2 growing to volume 8.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        cube = Cube(side_length=2, fill_opacity=0.5, color=BLUE)
        # Fixing per issue 28/43: Use E3
        self.place_at_grid(cube, 'E3', scale_factor=0.6)
        label = Text("Side = 2, Volume = 8", font_size=24).scale(0.75)
        label.next_to(cube, DOWN)
        self.play(Create(cube), Write(label))
        self.wait(2)
