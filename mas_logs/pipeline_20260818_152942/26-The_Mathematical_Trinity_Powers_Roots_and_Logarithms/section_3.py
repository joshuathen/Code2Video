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
        lecture_lines = ["Logarithms find the missing exponent.", "They tell us the time.", "How many generations to reach eight?"]
        self.setup_layout("Decoding the Logarithm: Finding the Time", lecture_lines)
        
        # Assets
        bacteria = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bacteria.svg")
        
        # Elements
        growth = VGroup(
            MathTex(r"2^0 = 1", color=WHITE),
            MathTex(r"2^1 = 2", color=WHITE),
            MathTex(r"2^2 = 4", color=WHITE),
            MathTex(r"2^3 = 8", color=WHITE),
            bacteria.copy().scale(0.5)
        ).arrange(DOWN)
        
        question = Text("How long (generations)?", color="#FFD700")
        answer = MathTex(r"\log_2(8) = 3", color="#32CD32")
        
        # Incorporate bacteria asset into answer
        answer_group = VGroup(answer, bacteria.copy().scale(0.5)).arrange(RIGHT)

        # === Animation for Lecture Line 1 ===
        self.place_in_area(growth, "B3", "C5", scale_factor=0.6)
        self.play(FadeIn(growth))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(question, "E4", scale_factor=0.7)
        self.play(Write(question))
        self.lecture[1].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(answer_group, "F4", scale_factor=1.0)
        self.play(FadeIn(answer_group))
        self.lecture[2].set_color("#32CD32")
        self.wait(2)
