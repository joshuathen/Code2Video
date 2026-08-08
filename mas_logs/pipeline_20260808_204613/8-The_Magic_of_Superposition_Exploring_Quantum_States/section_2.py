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
        lecture_lines = [
            "A qubit is a weighted sum: |ψ⟩ = α|0⟩ + β|1⟩.",
            "Think of a spinning, blurry coin.",
            "It is a blend of basis states.",
            "Represented as a vector on a plane.",
            "Coefficients determine the state's orientation."
        ]
        self.setup_layout("The Concept of Superposition", lecture_lines)
        
        # Colors for matching lecture lines
        colors = [BLUE, GREEN, YELLOW, RED, ORANGE]

        # === Animation for Lecture Line 1 ===
        psi = MathTex(r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle", color=colors[0])
        self.place_in_area(psi, 'A2', 'B5', scale_factor=0.9)
        self.play(Write(psi))
        self.lecture[0].set_color(colors[0])

        # === Animation for Lecture Line 2 ===
        # Use asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color=colors[1])
        coin_label = Text("Spinning Coin", font_size=20, color=WHITE).next_to(coin, DOWN)
        coin_group = VGroup(coin, coin_label)
        self.place_at_grid(coin_group, 'C2', scale_factor=0.8)
        self.play(FadeIn(coin), Write(coin_label))
        self.lecture[1].set_color(colors[1])

        # === Animation for Lecture Line 3 ===
        blend_text = Text("Blend of states", font_size=24, color=colors[2])
        self.place_at_grid(blend_text, 'C3', scale_factor=0.9)
        self.play(FadeIn(blend_text))
        self.lecture[2].set_color(colors[2])

        # === Animation for Lecture Line 4 ===
        vector = Arrow(ORIGIN, RIGHT * 1.5, color=colors[3], buff=0)
        plane = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_numbers": False}).scale(0.5)
        vec_group = VGroup(plane, vector)
        self.place_in_area(vec_group, 'D2', 'F5', scale_factor=0.6)
        self.play(Create(plane), GrowArrow(vector))
        self.lecture[3].set_color(colors[3])

        # === Animation for Lecture Line 5 ===
        coeff_text = Text("Orientation depends on α, β", font_size=20, color=colors[4])
        self.place_at_grid(coeff_text, 'F3', scale_factor=0.7)
        self.play(Write(coeff_text))
        self.lecture[4].set_color(colors[4])
        
        self.wait(2)
