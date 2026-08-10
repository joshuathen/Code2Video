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
        self.setup_layout("The Foundation: Exponential Growth", [
            "Exponential growth is repeated multiplication.",
            "Think of a rabbit population doubling.",
            "We represent this as two power x equals y.",
            "One robot makes two more every hour.",
            "After three hours, two cubed equals eight."
        ])
        
        # === Animation for Lecture Line 1 ===
        growth_eq = MathTex(r"b^x = y", color=WHITE)
        self.place_at_grid(growth_eq, "B4", scale_factor=1.0)
        self.play(Write(growth_eq))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/rabbit.svg]
        rabbit = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rabbit.svg")
        self.place_at_grid(rabbit, "D2", scale_factor=0.8)
        self.play(FadeIn(rabbit))
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.play(Indicate(growth_eq))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot, "D5", scale_factor=0.8)
        self.play(FadeIn(robot))
        self.lecture[3].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        calc = MathTex(r"2", r"^3", r" = 8", color=WHITE)
        calc[0].set_color(GOLD)
        calc[1].set_color(GOLD)
        self.place_at_grid(calc, "B6", scale_factor=1.0)
        self.play(Write(calc))
        self.lecture[4].set_color(YELLOW)
        self.play(Indicate(calc))
        self.wait(2)
