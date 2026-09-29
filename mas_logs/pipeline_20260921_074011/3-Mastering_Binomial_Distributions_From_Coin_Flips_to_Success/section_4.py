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
            "Calculate 3 successes in 5 trials.",
            "Changing p shifts graph center.",
            "Robot inspects microchips for defects.",
            "Probability changes with p value.",
            "Distribution reveals defect likelihood."
        ]
        self.setup_layout("Application: The Robot Quality Control", lecture_lines)
        
        # Assets
        robot_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"
        microchip_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/microchip.svg"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        robots = VGroup(*[SVGMobject(robot_svg, fill_opacity=1, fill_color=BLUE) for _ in range(5)])
        robots.arrange(RIGHT, buff=0.2)
        self.place_in_area(robots, 'D1', 'F2', scale_factor=0.6)
        self.play(FadeIn(robots))
        for i in range(3):
            self.play(robots[i].animate.set_color(GREEN), run_time=0.2)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        formula = MathTex(r"P(X=3) = \binom{5}{3} p^3 (1-p)^{5-3}", font_size=32)
        self.place_at_grid(formula, 'B4', scale_factor=1.0)
        self.play(Write(formula))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        microchip = SVGMobject(microchip_svg, fill_opacity=1, fill_color=ORANGE)
        self.place_at_grid(microchip, 'B6', scale_factor=0.5)
        self.play(FadeIn(microchip), Indicate(microchip))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        result = Text("P \approx 0.0729", font_size=32, color=GREEN)
        self.place_at_grid(result, 'E4', scale_factor=0.9)
        self.play(Write(result))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        bar_chart = BarChart(values=[0.1, 0.3, 0.4, 0.15, 0.05], bar_colors=[BLUE], y_range=[0, 0.5, 0.1], x_length=3, y_length=2)
        self.place_at_grid(bar_chart, 'D6', scale_factor=0.4)
        self.play(FadeIn(bar_chart), run_time=2)
        self.wait(1)
