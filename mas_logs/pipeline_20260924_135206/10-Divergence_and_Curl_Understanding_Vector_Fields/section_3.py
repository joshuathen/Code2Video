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
        lecture_lines_text = [
            "Curl measures local rotation behavior.",
            "A paddlewheel reveals angular motion.",
            "Whirlpools exhibit non-zero curl.",
            "It captures spinning tendencies.",
            "Rotation intensity is mathematically defined."
        ]
        self.setup_layout("Curl: The Rotation Factor", lecture_lines_text)

        # Pre-create objects
        paddle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/whirlpool.svg")
        field = ArrowVectorField(lambda p: np.array([-p[1], p[0], 0]), x_range=[-2, 2], y_range=[-2, 2])
        formula = MathTex(r"\nabla \times \mathbf{F}", color="#FFCC00")
        
        # Initial positioning
        self.place_at_grid(paddle, 'B4', scale_factor=0.6)
        self.place_at_grid(formula, 'E5', scale_factor=0.8)

        # === Animation for Lecture Line 1: Curl measures local rotation behavior. ===
        self.lecture[0].set_color("#00FF00")
        self.play(Create(field))
        self.wait(1)

        # === Animation for Lecture Line 2: A paddlewheel reveals angular motion. ===
        self.lecture[1].set_color("#00FF00")
        self.play(FadeIn(paddle))
        self.play(Rotate(paddle, angle=2*PI, rate_func=linear, run_time=2))
        self.wait(0.5)

        # === Animation for Lecture Line 3: Whirlpools exhibit non-zero curl. ===
        self.lecture[2].set_color("#00FF00")
        # Rotating animation is just a rotation wrapper, using Rotate is better
        self.play(Rotate(paddle, angle=2*PI, rate_func=linear, run_time=2))
        self.wait(0.5)

        # === Animation for Lecture Line 4: It captures spinning tendencies. ===
        self.lecture[3].set_color("#00FF00")
        self.play(Indicate(paddle))
        self.wait(0.5)

        # === Animation for Lecture Line 5: Rotation intensity is mathematically defined. ===
        self.lecture[4].set_color("#00FF00")
        self.play(Write(formula))
        self.wait(1)
