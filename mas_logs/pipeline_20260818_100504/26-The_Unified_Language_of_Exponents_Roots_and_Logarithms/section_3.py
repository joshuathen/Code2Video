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
            "Logarithms find the time.",
            "Solve for unknown exponents.",
            "Log base three of nine is two.",
            "What power raises three to nine?",
            "The squirrel finds the answer."
        ]
        self.setup_layout("Logarithms: Finding the Hidden Time", lecture_lines)
        
        # Mobjects
        eq_exp = MathTex("3^x = 9").scale(1.2)
        eq_log = MathTex(r"\log_3(9) = x").scale(1.2)
        
        # Load Asset
        squirrel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/squirrel.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        self.place_at_grid(squirrel, "A5", scale_factor=0.3)
        self.play(FadeIn(squirrel))
        self.place_in_area(eq_exp, 'B4', 'B6', scale_factor=1.0)
        self.play(Write(eq_exp))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.place_in_area(eq_log, 'C4', 'C6', scale_factor=1.0)
        self.play(Write(eq_log))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        # Highlighting the base
        base_three = MathTex("3", color="#00FF00").scale(1.2).move_to(eq_log[0][1])
        self.play(Indicate(base_three))
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FFFF")
        self.play(FadeOut(eq_exp))
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF00FF")
        final_answer = MathTex("x = 2").scale(1.5).set_color("#FF00FF")
        self.place_at_grid(final_answer, 'E5', scale_factor=1.2)
        self.play(Write(final_answer))
        # Move squirrel to final position
        self.play(squirrel.animate.move_to(self.grid["E2"]))
        self.wait(2)
