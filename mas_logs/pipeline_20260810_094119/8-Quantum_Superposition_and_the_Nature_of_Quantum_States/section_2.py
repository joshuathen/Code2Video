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
        self.setup_layout("Visualizing Superposition", [
            "We use the Bloch Sphere visualization.",
            "Quantum states are vectors on it.",
            "Superposition is defined by alpha and beta.",
            "Think of a spinning coin blur.",
            "Vectors represent $|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle$."
        ])
        
        # Assets
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        sphere = Sphere(radius=1.5, fill_opacity=0.2, color=WHITE).set_stroke(WHITE, opacity=0.5)
        axes = ThreeDAxes(x_range=[-1.5, 1.5], y_range=[-1.5, 1.5], z_range=[-1.5, 1.5], axis_config={"include_tip": True})
        bloch_group = VGroup(sphere, axes)
        # Applying critique fixes for position
        self.place_in_area(bloch_group, 'A3', 'D6', scale_factor=0.5)
        self.play(Create(bloch_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        vector = Arrow(start=ORIGIN, end=np.array([1, 0, 1]), buff=0, color=GOLD)
        # Adjust vector position to match sphere
        vector.move_to(bloch_group.get_center())
        label = MathTex(r"|\psi\rangle", color=GOLD).next_to(vector.get_end(), UP, buff=0.1)
        
        coin_copy = coin.copy()
        self.place_at_grid(coin_copy, 'B2', scale_factor=0.3)
        
        self.add(vector, label)
        self.play(GrowArrow(vector), Write(label), FadeIn(coin_copy))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00CED1")
        formula = MathTex(r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle", color="#00FFFF")
        # Fix: E3 to prevent crowding
        self.place_at_grid(formula, 'E3', scale_factor=0.9)
        self.play(Write(formula))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFD700")
        self.play(Rotate(vector, angle=2*PI, about_point=bloch_group.get_center(), run_time=2))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        coin_copy_2 = coin.copy()
        self.place_at_grid(coin_copy_2, 'E5', scale_factor=0.3)
        coeff_highlight = SurroundingRectangle(formula[0][4:10], color=YELLOW)
        self.play(Create(coeff_highlight), FadeIn(coin_copy_2))
        self.wait(2)
