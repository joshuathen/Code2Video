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
        lecture_lines = ["Viscosity eventually dissipates kinetic energy.", "Kolmogorov length scale limits eddy size.", "Structure breaks down into heat."]
        self.setup_layout("The Kolmogorov Length Scale", lecture_lines)
        
        # Formula for Kolmogorov length scale
        formula = MathTex(r"\eta = \left( \frac{\nu^3}{\epsilon} \right)^{1/4}", color=WHITE)
        self.place_in_area(formula, 'B1', 'B4', scale_factor=0.6)
        
        # Visualize Eddies
        eddies = VGroup(
            Circle(radius=1.5, color=BLUE, stroke_width=2),
            Circle(radius=0.75, color=BLUE, stroke_width=2),
            Circle(radius=0.3, color=BLUE, stroke_width=2),
            Circle(radius=0.1, color=RED, stroke_width=4)
        ).arrange(RIGHT, buff=0.2)
        
        # Add heat icon
        heat_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/heat.svg", color=RED)
        heat_icon.next_to(eddies[3], RIGHT, buff=0.1)
        eddies.add(heat_icon)
        
        self.place_at_grid(eddies, 'E5', scale_factor=0.45)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeIn(eddies[0:3]))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(FadeIn(eddies[3]), FadeIn(heat_icon))
        self.play(Indicate(eddies[3]))
        self.wait(2)
