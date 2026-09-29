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
            "The Koch Snowflake starts from a single segment.",
            "We replace one line with four smaller segments.",
            "Each segment is one-third the original length.",
            "This recursive process occupies more than 1D space.",
            "Fractal dimension D equals log 4 over log 3."
        ]
        self.setup_layout("Calculating the Fractal Dimension: The Koch Snowflake", lecture_lines)
        
        # Animation Elements
        line = Line(start=self.grid["C2"], end=self.grid["C5"], color=GOLD)
        snowflake = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/snowflake.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]), Create(line))
        self.play(self.lecture[0].animate.set_color(GOLD))

        # === Animation for Lecture Line 2 ===
        # Representing the 1 to 4 replacement
        new_segments = VGroup(
            Line(self.grid["C2"], self.grid["D3"]),
            Line(self.grid["D3"], self.grid["B3"]),
            Line(self.grid["B3"], self.grid["D4"]),
            Line(self.grid["D4"], self.grid["C5"])
        ).set_color(BLUE)
        
        # Fix: Positioning the animation in the grid area B4-E6 (Issue 27)
        self.place_in_area(snowflake, 'B4', 'E6', scale_factor=0.9)
        
        self.play(FadeIn(self.lecture[1]), FadeOut(line), Create(new_segments), FadeIn(snowflake))
        self.play(self.lecture[1].animate.set_color(BLUE))

        # === Animation for Lecture Line 3 ===
        one_third_label = Text("1/3", font_size=20, color=RED)
        # Fix: Position label at D3 (Issue 29)
        self.place_at_grid(one_third_label, 'D3', scale_factor=0.7)
        
        self.play(FadeIn(self.lecture[2]), Write(one_third_label))
        self.play(self.lecture[2].animate.set_color(RED))

        # === Animation for Lecture Line 4 ===
        self.play(FadeIn(self.lecture[3]))
        self.play(self.lecture[3].animate.set_color(PURPLE))

        # === Animation for Lecture Line 5 ===
        formula = MathTex(r"D = \frac{\log(4)}{\log(3)} \approx 1.26", color=ORANGE)
        # Fix: Position formula at D4 (Issue 28)
        self.place_at_grid(formula, 'D4', scale_factor=1.0)
        
        self.play(FadeIn(self.lecture[4]), Write(formula))
        self.play(self.lecture[4].animate.set_color(ORANGE))
        
        self.wait(2)
