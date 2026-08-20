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
        lecture_lines = ["The Wave Equation tracks rhythmic oscillations.", "Local disturbances trigger wave propagation.", "Speed depends on the material's properties."]
        self.setup_layout("The Wave Equation: Ripples and Rhythm", lecture_lines)
        
        # Elements
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg
        string = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg", color=WHITE)
        string_group = VGroup(string)
        # Using recommendation from issue 43
        self.place_in_area(string_group, 'B4', 'E6', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        # Draw a string at equilibrium position.
        self.play(Create(string_group), self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate string displacement u(x, t) based on [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg]
        displacement = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg", color="#0000FF")
        # Using recommendation from issue 42
        self.place_at_grid(displacement, 'D5', scale_factor=0.7)
        self.play(ReplacementTransform(string_group, displacement), self.lecture[1].animate.set_color("#0000FF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visualize wave propagation along the string. Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg
        wave = displacement.copy()
        wave.set_color("#FFFF00")
        self.play(self.lecture[2].animate.set_color("#FFFF00"), FadeIn(wave))
        self.play(wave.animate.shift(0.5 * RIGHT), run_time=2)
        self.wait(1)
