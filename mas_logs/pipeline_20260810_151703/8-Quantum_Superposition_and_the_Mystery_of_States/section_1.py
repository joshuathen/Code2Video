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
        lecture_lines = [
            "Classical bits are strictly zero or one.",
            "Quantum qubits exist in multiple configurations simultaneously.",
            "Think of a spinning, blurred coin.",
            "It represents both states at once.",
            "Observation forces a single definite result."
        ]
        self.setup_layout("The Classical vs. Quantum Prelude", lecture_lines)
        
        # Assets
        ball_img = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg"
        
        # Define objects
        # Using SVGMobject for asset integration as per instruction
        ball_0 = SVGMobject(ball_img, color=BLUE).scale(0.5)
        ball_1 = SVGMobject(ball_img, color=RED).scale(0.5)
        label_0 = MathTex(r"|0\rangle").next_to(ball_0, UP)
        label_1 = MathTex(r"|1\rangle").next_to(ball_1, UP)
        classical_group = VGroup(ball_0, ball_1, label_0, label_1).arrange(RIGHT, buff=0.5)
        
        psi_vec = Arrow(start=ORIGIN, end=UP*1.5, color=YELLOW)
        label_psi = MathTex(r"|\psi\rangle").next_to(psi_vec, RIGHT)
        quantum_group = VGroup(psi_vec, label_psi)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(classical_group, 'B4', scale_factor=0.7)
        self.play(Create(classical_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(quantum_group, 'B6', scale_factor=0.7)
        self.play(GrowArrow(psi_vec), Write(label_psi))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(ORANGE))
        coin = Circle(radius=0.5, color=GREY, fill_opacity=0.5)
        self.place_at_grid(coin, 'D5', scale_factor=0.8)
        self.play(FadeIn(coin))
        self.play(Rotate(coin, angle=2*PI, run_time=1))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(GREEN))
        self.play(coin.animate.set_color(PURPLE))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(RED))
        self.play(FadeOut(psi_vec, label_psi, coin))
        self.play(Indicate(ball_0))
