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
            "Fractal dimension quantifies the space-filling capacity.",
            "Calculated as the log of copies over log scale.",
            "The Koch snowflake has a dimension of 1.26.",
            "Non-integer values represent roughness or complexity.",
            "Fractal dimension captures detail at every scale."
        ]
        self.setup_layout("Defining Fractal Dimension (Hausdorff Dimension)", lecture_lines)
        
        # Elements
        line = Line(LEFT, RIGHT)
        segment = Line(LEFT, RIGHT).scale(1/3)
        four_segments = VGroup(*[segment.copy() for _ in range(4)]).arrange(RIGHT, buff=0)
        snowflake = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/snowflake.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_at_grid(line, 'C3', scale_factor=0.9)
        self.play(Create(line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        eqn = MathTex(r"D = \frac{\log(4)}{\log(3)}").set_color("#00FF00")
        self.place_at_grid(eqn, 'D4', scale_factor=1.1)
        # Using snowflake asset instead of line segments per storyboard/issue 18
        self.place_at_grid(snowflake, 'C3', scale_factor=0.5)
        self.play(Transform(line, snowflake))
        self.play(Write(eqn))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(BLUE)
        result = MathTex(r"D \approx 1.26").set_color("#FFCC00")
        self.place_at_grid(result, 'E5', scale_factor=1.2)
        self.play(FadeIn(result))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(BLUE)
        self.wait(1)
