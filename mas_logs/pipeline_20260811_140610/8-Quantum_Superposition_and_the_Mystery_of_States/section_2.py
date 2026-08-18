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
            "A qubit can be in two states simultaneously.",
            "Represented by |ψ⟩ = α|0⟩ + β|1⟩.",
            "Superposition is like playing a chord.",
            "It combines multiple states together.",
            "Measure to 'hear' one state."
        ])

        guitar_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/guitar.svg", color=WHITE)
        state_0 = MathTex(r"|0\rangle", color=WHITE)
        state_1 = MathTex(r"|1\rangle", color=WHITE)
        eq = MathTex(r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(guitar_icon, "B4", scale_factor=0.5)
        self.place_at_grid(state_0, "B2", scale_factor=1.0)
        self.place_at_grid(state_1, "B6", scale_factor=1.0)
        self.play(FadeIn(state_0), FadeIn(state_1), FadeIn(guitar_icon))
        self.lecture[0].set_color("#FF9900")

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(eq, "D4", scale_factor=1.0)
        self.play(Write(eq))
        self.lecture[1].set_color("#00FF99")

        # === Animation for Lecture Line 3 ===
        point = Dot(self.grid["E2"], color="#FF00FF")
        self.play(FadeIn(point))
        self.play(point.animate.move_to(self.grid["E6"]), run_time=1.5)
        self.play(point.animate.move_to(self.grid["E2"]), run_time=1.5)
        self.lecture[2].set_color("#FF00FF")

        # === Animation for Lecture Line 4 ===
        blurred = VGroup(state_0.copy(), state_1.copy()).set_color("#00FFFF")
        self.place_at_grid(blurred, "E4", scale_factor=1.2)
        self.play(FadeIn(blurred))
        self.lecture[3].set_color("#00FFFF")

        # === Animation for Lecture Line 5 ===
        final_state = MathTex(r"|0\rangle", color=WHITE)
        self.place_at_grid(final_state, "F4", scale_factor=1.0)
        self.play(FadeOut(blurred), FadeOut(point), FadeIn(final_state), guitar_icon.animate.set_color(WHITE))
        self.lecture[4].set_color("#FFFFFF")
        self.wait(1)
