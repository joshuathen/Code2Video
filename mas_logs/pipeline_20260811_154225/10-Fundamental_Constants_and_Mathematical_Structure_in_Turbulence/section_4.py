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
            "The Kolmogorov constant is universal across fluids.",
            "Epsilon governs the rate of energy dissipation.",
            "Cascade structure remains consistent regardless of geometry.",
            "Octopus and drone wakes share spectral slopes.",
            "Physical laws transcend specific environmental scales."
        ]
        self.setup_layout("The Universal Nature of the Constants", lecture_lines)
        
        # Assets
        fan = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fan.svg")
        jet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/jet.svg")
        fan_label = Text("Experiment A", font_size=16)
        jet_label = Text("Experiment B", font_size=16)
        
        C_label = MathTex(r"C", color="#00FFFF").scale(2)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.place_at_grid(fan, "B1", scale_factor=0.8)
        self.place_at_grid(jet, "C5", scale_factor=0.8)
        fan_label.next_to(fan, UP, buff=0.1)
        jet_label.next_to(jet, UP, buff=0.1)
        self.play(FadeIn(fan), FadeIn(jet), Write(fan_label), Write(jet_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        epsilon = MathTex(r"\epsilon", color="#FF00FF").scale(1.5)
        self.place_at_grid(epsilon, "E2", scale_factor=0.7)
        self.play(Write(epsilon))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        line = Line(fan.get_bottom(), jet.get_bottom(), color="#FFFF00")
        self.play(Create(line))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        self.place_at_grid(C_label, "D4", scale_factor=0.6)
        self.play(FadeIn(C_label))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        self.play(FadeOut(fan), FadeOut(jet), FadeOut(fan_label), FadeOut(jet_label), FadeOut(epsilon), FadeOut(line))
        self.play(C_label.animate.scale(0.8).move_to(self.grid["C3"]))
