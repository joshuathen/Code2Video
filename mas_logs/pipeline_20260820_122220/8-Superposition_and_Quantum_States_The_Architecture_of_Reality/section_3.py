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
            "Measurement causes the wavefunction to collapse.",
            "The system settles into a single basis state.",
            "Measurement probabilities are squares of complex amplitudes."
        ]
        self.setup_layout("The Measurement Problem", lecture_lines)
        
        # Animation Elements
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/voltmeter.svg]
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg]
        
        wavefunction_circle = Circle(radius=0.5, color=BLUE)
        self.place_at_grid(wavefunction_circle, 'B5', scale_factor=0.6)
        
        state_vec = Arrow(start=wavefunction_circle.get_center(), end=wavefunction_circle.get_center()+UP*0.5, color=YELLOW)
        state_label = MathTex(r"|\psi\rangle").scale(0.7)
        self.place_at_grid(state_label, 'B4', scale_factor=0.7)
        
        apparatus = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/voltmeter.svg", color=WHITE)
        self.place_at_grid(apparatus, 'C5', scale_factor=0.5)
        
        basis_0 = MathTex(r"|0\rangle").scale(0.8)
        basis_1 = MathTex(r"|1\rangle").scale(0.8)
        
        sensor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg", color=RED)
        self.place_at_grid(sensor_icon, 'F5', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(wavefunction_circle), GrowArrow(state_vec), FadeIn(apparatus))
        self.play(apparatus.animate.shift(UP*0.5), run_time=1.5)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        self.play(
            state_vec.animate.rotate(PI/2, about_point=wavefunction_circle.get_center()).set_color(GREEN),
            FadeIn(basis_0),
            FadeIn(sensor_icon)
        )
        self.play(Indicate(basis_0))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        prob_eqn = MathTex(r"P(|0\rangle) = |\alpha|^2").scale(0.8)
        self.place_at_grid(prob_eqn, 'E5', scale_factor=0.9)
        self.play(Write(prob_eqn))
        self.wait(2)
