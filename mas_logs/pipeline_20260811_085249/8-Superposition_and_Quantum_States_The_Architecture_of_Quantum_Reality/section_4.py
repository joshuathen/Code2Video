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
            "Measurement forces a collapse to basis states.",
            "The Born rule calculates the probability.",
            "Opening the box determines the outcome."
        ]
        self.setup_layout("The Collapse: Measurement and Reality", lecture_lines)
        
        # Load asset
        box = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg")
        self.place_at_grid(box, "D4", scale_factor=0.5)
        
        # Define visual elements
        equation = MathTex(r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle", substrings_to_isolate=[r"|", r"\rangle"])
        self.place_in_area(equation, "A1", "A6", scale_factor=0.9)
        
        state_vec = Vector([0.5, 0.5], color=BLUE)
        self.place_at_grid(state_vec, "D4", scale_factor=0.8)
        
        basis_0 = MathTex(r"|0\rangle")
        self.place_at_grid(basis_0, "B3", scale_factor=1.0)
        
        basis_1 = MathTex(r"|1\rangle")
        self.place_at_grid(basis_1, "D3", scale_factor=1.0)
        
        arrow = Arrow(start=UP, end=DOWN, color=YELLOW)
        self.place_at_grid(arrow, "C2", scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        # Display state vector |ψ⟩ stored within a box before measurement.
        self.play(FadeIn(box), Create(state_vec), Write(equation))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate measurement operation on state vector.
        self.play(FadeIn(basis_0), FadeIn(basis_1), GrowArrow(arrow))
        self.lecture[1].set_color(GREEN)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show vector collapse to |0⟩ or |1⟩ in #FF0000.
        collapse_anim = state_vec.animate.set_color("#FF0000").move_to(self.grid["B3"])
        self.play(collapse_anim)
        self.lecture[2].set_color("#FF0000")
        self.wait(2)
