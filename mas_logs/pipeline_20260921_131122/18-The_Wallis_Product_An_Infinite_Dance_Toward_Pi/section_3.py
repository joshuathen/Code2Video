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
            "The Wallis product formula emerges here.",
            "Fractions multiply towards pi over two.",
            "Wallis-Bot assembles this infinite product.",
            "Output values converge to pi halves.",
            "Infinite steps reveal the target value."
        ]
        self.setup_layout("The Wallis Product Formula", lecture_lines)
        
        # Math objects
        wallis_expr = MathTex(
            r"{\pi \over 2} = \prod_{n=1}^{\infty} \left( {2n \over 2n-1} \cdot {2n \over 2n+1} \right)",
            font_size=36
        )
        self.place_in_area(wallis_expr, 'A3', 'C6', scale_factor=0.6)
        
        # Asset for robot
        robot_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"
        bot = SVGMobject(robot_asset, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Write(wallis_expr))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        fraction_group = MathTex(
            r"\left( {2 \over 1} \cdot {2 \over 3} \right) \left( {4 \over 3} \cdot {4 \over 5} \right) \cdots",
            font_size=30
        )
        self.place_at_grid(fraction_group, 'D5', scale_factor=0.7)
        self.play(Write(fraction_group))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.place_at_grid(bot, 'E4', scale_factor=0.8)
        self.play(FadeIn(bot))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(ORANGE))
        display = DecimalNumber(1.0, num_decimal_places=4, color=WHITE)
        self.place_at_grid(display, 'F4', scale_factor=0.8)
        self.play(Write(display))
        self.play(display.animate.set_value(1.5707))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        self.play(Indicate(wallis_expr))
        self.wait(1)
