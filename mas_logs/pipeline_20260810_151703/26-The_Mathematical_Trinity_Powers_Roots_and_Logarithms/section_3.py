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
            "Logarithms answer how much time passed.",
            "They find the exponent, the unknown time.",
            "If three to x is eighty-one, x is four.",
            "The logarithm of eighty-one base three is four.",
            "Think of it like reading a treasure map."
        ]
        self.setup_layout("The 'Hidden' Question: Logarithms", lecture_lines)
        
        # Colors
        COLOR_HIGHLIGHT = YELLOW
        COLOR_FINAL = "#00FF00"
        
        # Load Assets
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        treasure_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/treasure.svg")

        # === Animation for Lecture Line 1 ===
        formula = MathTex(r"\\log_{b}(y) = x", color=WHITE)
        self.place_in_area(formula, 'A2', 'C4', scale_factor=0.7)
        self.play(Write(formula))
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(map_icon, 'D3', scale_factor=0.8)
        self.play(FadeIn(map_icon))
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        formula_case = MathTex(r"3", r"^x", "=", r"81")
        self.place_in_area(formula_case, 'A2', 'C4', scale_factor=0.8)
        self.play(ReplacementTransform(formula, formula_case))
        self.place_at_grid(treasure_icon, 'D4', scale_factor=0.8)
        self.play(FadeIn(treasure_icon))
        
        base_3 = formula_case[0]
        result_81 = formula_case[3]
        
        self.play(base_3.animate.set_color(RED), result_81.animate.set_color(BLUE))
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        pointer = Arrow(start=map_icon.get_center(), end=treasure_icon.get_center(), color=WHITE)
        self.play(Create(pointer))
        self.lecture[3].set_color(COLOR_HIGHLIGHT)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        final_eq = MathTex(r"\\log_{3}(81) = 4", color=COLOR_FINAL)
        self.place_in_area(final_eq, 'D5', 'F6', scale_factor=0.9)
        self.play(FadeOut(formula_case), FadeOut(pointer), FadeOut(map_icon), FadeOut(treasure_icon), FadeIn(final_eq))
        self.lecture[4].set_color(COLOR_HIGHLIGHT)
        self.wait(2)
