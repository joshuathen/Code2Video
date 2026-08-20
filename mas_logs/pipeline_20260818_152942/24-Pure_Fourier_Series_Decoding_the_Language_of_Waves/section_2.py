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
        self.setup_layout("The Mathematical Foundation: Basis Functions", [
            "Fourier series defines signals by basis functions.",
            "Sine and cosine waves act as independent axes.",
            "They combine to reconstruct any complex wave."
        ])
        
        # Load assets
        asset1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        asset2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        # === Animation for Lecture Line 1 ===
        # Draw sine wave #FFFFFF with label f1
        sine_wave = FunctionGraph(lambda x: 0.5 * np.sin(x * PI), x_range=[-2, 2], color="#FFFFFF")
        f1_label = Text("f1", color="#FFFFFF", font_size=20)
        self.place_at_grid(sine_wave, "A3", scale_factor=0.8)
        self.place_at_grid(f1_label, "A2", scale_factor=0.7)
        self.place_at_grid(asset1, "A1", scale_factor=0.5)
        self.play(Create(sine_wave), Write(f1_label), FadeIn(asset1))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Draw higher frequency wave #FF00FF with label f2
        high_wave = FunctionGraph(lambda x: 0.5 * np.sin(x * 2 * PI), x_range=[-2, 2], color="#FF00FF")
        f2_label = Text("f2", color="#FF00FF", font_size=20)
        self.place_at_grid(high_wave, "C3", scale_factor=0.8)
        self.place_at_grid(f2_label, "C2", scale_factor=0.7)
        self.play(Create(high_wave), Write(f2_label))
        self.lecture[1].set_color("#FF00FF")
        
        # === Animation for Lecture Line 3 ===
        # Display summation formula #00FFFF for basis functions
        formula = MathTex(r"f(t) = \sum a_n \sin(nt) + b_n \cos(nt)", color="#00FFFF")
        self.place_in_area(formula, "E2", "E5", scale_factor=0.6)
        self.place_at_grid(asset2, "F5", scale_factor=0.5)
        self.play(Write(formula), FadeIn(asset2))
        self.lecture[2].set_color("#00FFFF")
        
        self.wait(2)
