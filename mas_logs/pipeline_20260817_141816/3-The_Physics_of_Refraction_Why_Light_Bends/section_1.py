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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Nature of Light Waves", [
            "Light travels as a constant wave in a vacuum.",
            "Optical density measures how a medium slows light down.",
            "Denser materials cause light to slow down significantly."
        ])
        
        # Assets
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        vacuum = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vacuum.svg")
        
        wave = FunctionGraph(lambda x: 0.5 * np.sin(4 * x), x_range=[-PI, PI], color=WHITE)
        self.place_in_area(wave, 'B4', 'D6', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(prism, 'B2', scale_factor=0.5)
        self.play(FadeIn(prism), Create(wave), run_time=2)
        self.play(self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 2 ===
        horizontal_arrow = DoubleArrow(start=self.grid['E2'], end=self.grid['E4'], color=BLUE, buff=0.1)
        self.place_at_grid(horizontal_arrow, 'E3', scale_factor=0.6)
        self.play(Create(horizontal_arrow))
        self.play(self.lecture[1].animate.set_color(BLUE))

        # === Animation for Lecture Line 3 ===
        vertical_arrows = VGroup(
            DoubleArrow(start=self.grid['C5'] + UP*0.2, end=self.grid['C5'] + DOWN*0.2, color=RED, buff=0.1),
        )
        self.place_at_grid(vertical_arrows, 'C5', scale_factor=0.5)
        self.place_at_grid(vacuum, 'F5', scale_factor=0.5)
        
        self.play(Create(vertical_arrows), FadeIn(vacuum))
        self.play(self.lecture[2].animate.set_color(RED))
        
        self.wait(2)
