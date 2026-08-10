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
            "The Kolmogorov constant (C) is approximately 1.5.",
            "Turbulence structure is statistically universal.",
            "Physics remains independent of specific fluid conditions.",
            "Clouds and nuclear coolants share spectral slopes.",
            "Universality simplifies modeling complex turbulent flows."
        ]
        self.setup_layout("The Universal Constant: Kolmogorov Constant (C)", lecture_lines)
        
        # Assets
        cloud_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cloud.svg")
        coolant_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coolant.svg")
        
        # Elements
        C_tex = MathTex("C \\approx 1.5", font_size=48, color=BLUE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_at_grid(cloud_icon, 'B2', scale_factor=0.5)
        self.play(FadeIn(cloud_icon))
        # Following fix from issue 47: use C5 instead of C3, and 49: use D4 for the formula.
        self.place_at_grid(C_tex, 'D4', scale_factor=1.0)
        self.play(FadeIn(C_tex))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(ORANGE)
        self.place_at_grid(coolant_icon, 'E5', scale_factor=0.5)
        self.play(FadeIn(coolant_icon))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(RED)
        self.wait(1)
