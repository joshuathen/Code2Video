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
        lecture_lines = [
            "Differentiation and integration are inverse machines.",
            "Derivatives map a function to change.",
            "Integration maps change back to the original.",
            "Constant C handles the starting point.",
            "These operations essentially cancel each other."
        ]
        self.setup_layout("Visual Synthesis: Inverse Operations", lecture_lines)
        
        # Define machine boxes using SVG asset
        machine_diff = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg", color=BLUE)
        label_diff = Text("Derivative", font_size=18).next_to(machine_diff, UP, buff=0.1)
        
        machine_int = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg", color=GREEN)
        label_int = Text("Integral", font_size=18).next_to(machine_int, UP, buff=0.1)

        # Apply layout fixes from critique
        self.place_at_grid(machine_diff, 'B2', scale_factor=0.6)
        self.add(label_diff)
        self.place_at_grid(machine_int, 'E2', scale_factor=0.6)
        self.add(label_int)

        func_f = MathTex(r"f(x) = x^2", font_size=24)
        func_df = MathTex(r"f'(x) = 2x", font_size=24)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(machine_diff), FadeIn(label_diff), FadeIn(machine_int), FadeIn(label_int))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FF00FF"))
        self.place_at_grid(func_f, 'A2', scale_factor=0.6)
        self.play(FadeIn(func_f))
        self.play(func_f.animate.move_to(machine_diff.get_center()))
        self.play(FadeOut(func_f), FadeIn(func_df))
        self.place_at_grid(func_df, 'B4', scale_factor=0.7)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FF00FF"))
        self.play(func_df.animate.move_to(machine_int.get_center()))
        self.play(FadeOut(func_df), FadeIn(func_f))
        self.place_in_area(func_f, 'E3', 'E5', scale_factor=0.7)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(ORANGE))
        plus_c = MathTex(r"+ C", font_size=24, color=ORANGE)
        self.place_at_grid(plus_c, 'E6', scale_factor=0.7)
        self.play(FadeIn(plus_c))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color("#FF00FF"))
        self.play(Indicate(machine_diff, color="#FF00FF"), Indicate(machine_int, color="#FF00FF"))
