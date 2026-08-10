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
        self.setup_layout("Synthesizing the Structure", [
            "Energy density follows E(k) = Cε^{2/3}k^{-5/3}.",
            "Wavenumber (k) defines scales of vortex motion.",
            "Engineers use this to predict turbulence noise."
        ])
        
        # Assets
        turbine = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/turbine.svg")
        microphone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg")
        
        # === Animation for Lecture Line 1 ===
        formula = MathTex(r"E(k) = C \cdot \varepsilon^{2/3} \cdot k^{-5/3}", color=BLUE)
        self.place_in_area(formula, 'A2', 'A5', scale_factor=0.8)
        self.play(Write(formula))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        k_line = NumberLine(x_range=[0, 10, 1], length=5, include_numbers=True)
        dot = Dot(color=YELLOW)
        dot_label = Tex("k", color=YELLOW).next_to(dot, UP)
        k_group = VGroup(k_line, dot, dot_label)
        
        self.place_in_area(k_group, 'B2', 'C5', scale_factor=0.7)
        self.play(Create(k_line), FadeIn(dot), Write(dot_label))
        self.play(dot.animate.shift(RIGHT * 3), run_time=2)
        
        # Placing turbine and microphone for energy/dissipation comparison
        self.place_at_grid(turbine, 'C2', scale_factor=0.6)
        self.place_at_grid(microphone, 'C5', scale_factor=0.6)
        self.play(FadeIn(turbine), FadeIn(microphone))
        
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        rect = RoundedRectangle(corner_radius=0.2, color=GREEN)
        noise_text = Text("Turbulence Noise Model", font_size=24, color=GREEN)
        noise_group = VGroup(rect, noise_text).arrange(DOWN)
        
        self.place_in_area(noise_group, 'D2', 'E5', scale_factor=0.6)
        self.play(Create(rect), Write(noise_text))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
