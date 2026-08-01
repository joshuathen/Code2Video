from manim import *
import numpy as np

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
            "Superposition is a linear combination of basis states.",
            "We write the state |ψ⟩ as alpha |0⟩ plus beta |1⟩.",
            "Coefficients represent the probability amplitudes of each state.",
            "Systems exist in both states simultaneously before measurement.",
            "Think of a spinning coin as a blurred superposition."
        ]
        self.setup_layout("The State Vector and Superposition", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display the formula |psi> = alpha|0> + beta|1> in the center in white #FFFFFF.
        self.lecture[0].set_color("#FFFF00")
        
        # formula components: 
        # 0:|, 1:\psi, 2:\rangle, 3:=, 4:\alpha, 5:|, 6:0, 7:\rangle, 8:+, 9:\beta, 10:|, 11:1, 12:\rangle
        formula = MathTex(
            "|", "\\psi", "\\rangle", "=", "\\alpha", "|", "0", "\\rangle", "+", "\\beta", "|", "1", "\\rangle",
            color="#FFFFFF", font_size=42
        )
        # Fix Issue 30: Move formula to Row A-B
        self.place_in_area(formula, 'A2', 'B5')
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We write the state |ψ⟩ as alpha |0⟩ plus beta |1⟩.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight alpha and beta as 'probability amplitudes' with a pulsing yellow #FFFF00 effect.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        
        alpha = formula[4]
        beta = formula[9]
        
        label_amp = Text("probability amplitudes", font_size=20, color="#FFFF00")
        # Fix Issue 30: Move label_amp to Row C
        self.place_in_area(label_amp, 'C2', 'C5')

        # Pulsing effect
        self.play(
            alpha.animate.set_color("#FFFF00").scale(1.2),
            beta.animate.set_color("#FFFF00").scale(1.2),
            FadeIn(label_amp)
        )
        self.play(
            alpha.animate.scale(1/1.2),
            beta.animate.scale(1/1.2)
        )
        self.wait(2)
        self.play(FadeOut(label_amp), alpha.animate.set_color(WHITE), beta.animate.set_color(WHITE))

        # === Animation for Lecture Line 4 ===
        # Show a spinning coin animation as a blurry grey #808080 disk to represent superposition.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFF00")
        
        coin_disk = Circle(radius=0.8, color="#808080", fill_opacity=0.3, stroke_width=2)
        blurry_lines = VGroup(*[
            Line(ORIGIN, RIGHT * 0.7, color="#808080", stroke_opacity=0.2).rotate(a) 
            for a in np.linspace(0, 2*PI, 16)
        ])
        coin_group = VGroup(coin_disk, blurry_lines)
        # Fix Issue 31: Move coin_group to bottom right E5-F6
        self.place_in_area(coin_group, 'E5', 'F6')
        
        self.play(FadeIn(coin_group))
        self.play(Rotate(coin_group, angle=2*PI, rate_func=linear), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Think of a spinning coin as a blurred superposition.
        # Replace the disk with a green #00FF00 vector |psi> that rotates slightly between |0> and |1>.
        # Label the vector |psi> as 'Superposition' in bold green #00FF00 text.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFF00")
        
        # Coordinate system components
        axis_h = Arrow(ORIGIN, RIGHT * 1.5, color=WHITE, buff=0)
        axis_v = Arrow(ORIGIN, UP * 1.5, color=WHITE, buff=0)
        l0 = MathTex("|0\\rangle", font_size=24).next_to(axis_h, RIGHT, buff=0.1)
        l1 = MathTex("|1\\rangle", font_size=24).next_to(axis_v, UP, buff=0.1)
        axes_subgroup = VGroup(axis_h, axis_v, l0, l1)
        
        # Vector and Label components
        norm_dir = normalize(RIGHT + UP)
        vector_psi = Arrow(ORIGIN, norm_dir * 1.2, color="#00FF00", buff=0)
        label_sup = Text("Superposition", font_size=24, color="#00FF00", weight=BOLD)
        label_sup.next_to(vector_psi.get_end(), UR, buff=0.1)
        
        # Group everything together to place it correctly in the grid
        final_viz = VGroup(axes_subgroup, vector_psi, label_sup)
        # Fix Issue 31: Move final_viz to bottom area D1-F4
        self.place_in_area(final_viz, 'D1', 'F4')
        
        # Sequential reveal
        self.play(
            FadeOut(coin_group),
            FadeIn(axes_subgroup),
            FadeIn(vector_psi)
        )
        
        # Rotating effect as requested
        start_pt = vector_psi.get_start()
        self.play(
            Rotate(vector_psi, angle=PI/6, about_point=start_pt, rate_func=there_and_back),
            run_time=0.8
        )
        self.play(
            Rotate(vector_psi, angle=-PI/6, about_point=start_pt, rate_func=there_and_back),
            run_time=0.8
        )
        
        self.play(Write(label_sup))
        self.wait(3)
        
        # Cleanup final lecture line highlight
        self.lecture[4].set_color(WHITE)
        self.wait(1)
