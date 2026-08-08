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
        self.setup_layout("Prerequisite: Orthogonality of Sinusoids", [
            "Sine and cosine functions form a signal basis.",
            "Orthogonality ensures these components are independent.",
            "This independence allows us to isolate frequencies."
        ])
        
        # Setup visual elements
        wave1 = FunctionGraph(lambda x: 0.5 * np.sin(2 * np.pi * x), x_range=[-1, 1], color=BLUE)
        wave2 = FunctionGraph(lambda x: 0.5 * np.sin(4 * np.pi * x), x_range=[-1, 1], color=RED)
        label1 = MathTex(r"f_1(t)", color=BLUE)
        label2 = MathTex(r"f_2(t)", color=RED)
        integral_eq = MathTex(r"\int_0^T f_1(t)f_2(t)dt = 0", font_size=32, color=YELLOW)
        
        # Assets
        oscillator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/oscillator.svg")
        speaker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speaker.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(wave1, 'A2', scale_factor=0.8)
        self.place_at_grid(label1, 'A1', scale_factor=0.7)
        self.place_at_grid(oscillator, 'B1', scale_factor=0.5)
        self.play(Create(wave1), Write(label1), FadeIn(oscillator))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED))
        self.place_at_grid(wave2, 'C2', scale_factor=0.8)
        self.place_at_grid(label2, 'C1', scale_factor=0.7)
        self.play(Create(wave2), Write(label2))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.place_at_grid(speaker, 'D1', scale_factor=0.5)
        self.place_in_area(integral_eq, 'E2', 'F5', scale_factor=1.0)
        self.play(FadeIn(speaker), Write(integral_eq))
        self.wait(2)
