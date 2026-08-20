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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Defining Superposition", [
            "Superposition allows multiple states simultaneously.",
            "Formally, the state is alpha-zero plus beta-one.",
            "Normalization ensures the total probability equals one.",
            "Imagine a coin spinning between heads and tails.",
            "It holds all potential outcomes at once."
        ])
        
        CYAN_COLOR = "#00FFFF"
        YELLOW_COLOR = "#FFFF00"
        RED_COLOR = "#FF0000"

        # Setup visualization
        axes = Axes(x_range=[-1.5, 1.5], y_range=[-1.5, 1.5], axis_config={"include_tip": True})
        vector = Vector([0.8, 0.6], color=YELLOW_COLOR)
        label_0 = MathTex(r"|0\rangle").next_to(axes.c2p(1, 0), RIGHT)
        label_1 = MathTex(r"|1\rangle").next_to(axes.c2p(0, 1), UP)
        
        state_group = VGroup(axes, vector, label_0, label_1)
        self.place_in_area(state_group, "A4", "D6", scale_factor=0.6)

        # Asset
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(CYAN_COLOR)
        self.play(Create(state_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(CYAN_COLOR)
        alpha_beta_text = MathTex(r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle", color=YELLOW_COLOR)
        self.place_at_grid(alpha_beta_text, 'E1', scale_factor=0.7)
        self.play(Write(alpha_beta_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(CYAN_COLOR)
        norm_text = MathTex(r"|\alpha|^2 + |\beta|^2 = 1", color=RED_COLOR)
        self.place_at_grid(norm_text, 'E2', scale_factor=0.7)
        self.play(Write(norm_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(CYAN_COLOR)
        self.place_in_area(coin, 'B4', 'D6', scale_factor=0.6)
        self.play(FadeIn(coin))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(CYAN_COLOR)
        self.play(Rotate(vector, angle=PI/4, about_point=axes.get_origin()), Rotate(coin, angle=2*PI))
        self.wait(2)
