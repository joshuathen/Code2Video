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
            "Energy constraints form a circular boundary.",
            "Collisions represent the arc length approximation.",
            "The geometry reflects Pi's hidden presence.",
            "Phase space angles track the collisions.",
            "Each bounce maps to a Pi digit."
        ]
        self.setup_layout("Connecting Physics to Pi", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Use billiard.svg for the billiard table (circle representation)
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg")
        label1 = Text("Energy Constraint", font_size=20, color="#00FFFF")
        self.place_in_area(circle, "A2", "C4", scale_factor=0.5)
        # B011: tether label via .next_to
        label1.next_to(circle, DOWN, buff=0.1)
        self.play(FadeIn(circle), Write(label1))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        path = Line(start=circle.get_top(), end=circle.get_right(), color="#FFFF00")
        self.play(Create(path), FadeIn(ball.scale(0.3).move_to(circle.get_top())))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        pi_sym = MathTex(r"\pi", color="#FF00FF", font_size=40)
        self.place_at_grid(pi_sym, "B5", scale_factor=0.6)
        self.play(Write(pi_sym))
        self.lecture[2].set_color("#FF00FF")

        # === Animation for Lecture Line 4 ===
        dot = Dot(color=YELLOW)
        dot.move_to(circle.get_center() + rotate_vector(RIGHT * circle.width / 2, PI/4))
        self.play(FadeIn(dot))
        self.lecture[3].set_color(YELLOW)

        # === Animation for Lecture Line 5 ===
        digits = VGroup(*[Text(d, font_size=24) for d in ["3", ".", "1", "4"]])
        digits.arrange(RIGHT)
        # Using ball.svg as visual tether for bounces
        bounce_ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg").scale(0.2)
        bounce_ball.next_to(digits, UP)
        content = VGroup(digits, bounce_ball)
        self.place_at_grid(content, "D5", scale_factor=0.7)
        self.play(FadeIn(content))
        self.lecture[4].set_color(WHITE)
        
        self.wait(2)
