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
        self.setup_layout("The Dissipation Scale: Where Chaos Ends", [
            "Kinetic energy dissipates at microscopic scales.",
            "This occurs at the Kolmogorov length scale.",
            "Viscous forces convert motion into thermal heat."
        ])
        
        # Create elements
        formula = MathTex(r"\\eta = (\\nu^3 / \\varepsilon)^{1/4}", color="#FF5733")
        eddy_group = VGroup(
            Circle(radius=0.5, color=BLUE),
            Circle(radius=0.3, color=BLUE).shift(UP*0.3),
            Circle(radius=0.2, color=BLUE).shift(RIGHT*0.3)
        )
        dissipation_mark = Dot(color="#FFFF00")
        
        # Assets
        heater = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/heater.svg")
        thermometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thermometer.svg")
        
        # === Animation for Lecture Line 1 ===
        # Kinetic energy dissipates at microscopic scales.
        self.lecture[0].set_color("#FFD700")
        self.place_in_area(eddy_group, 'B2', 'C4', scale_factor=0.7)
        self.play(Create(eddy_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This occurs at the Kolmogorov length scale.
        self.lecture[1].set_color("#FF5733")
        self.place_at_grid(formula, 'E5', scale_factor=0.9)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Viscous forces convert motion into thermal heat.
        self.lecture[2].set_color("#00FF00")
        self.place_at_grid(dissipation_mark, 'C6', scale_factor=0.6)
        
        self.place_at_grid(heater, 'D6', scale_factor=0.8)
        self.place_at_grid(thermometer, 'E6', scale_factor=0.8)
        
        self.play(
            Flash(dissipation_mark, color="#FF0000", line_length=0.2, num_lines=12),
            FadeIn(heater),
            FadeIn(thermometer)
        )
        self.wait(2)
