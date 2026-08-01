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
        # Data for setup
        title = "Prerequisite Check: The Derivative as a 'Slicer'"
        lines = [
            "Derivatives \"slice\" functions into tiny, instantaneous changes.",
            "They reveal the slope of a tangent line.",
            "In motion, the derivative of position is velocity."
        ]
        
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Derivatives "slice" functions into tiny, instantaneous changes.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # A smooth white curve (#FFFFFF)
        curve = ParametricFunction(
            lambda t: np.array([t, 0.25 * t**2, 0]),
            t_range=[-2, 2],
            color=WHITE
        )
        # Position curve centrally in the right-side area
        self.place_in_area(curve, 'B2', 'E5', scale_factor=1.2)
        
        self.play(Create(curve))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # They reveal the slope of a tangent line.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Magnifying glass asset integration (Issue 26)
        # Positioned at C4 with scale 1.3 (Issue 33)
        magnifier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        magnifier.set_color("#C0C0C0")
        self.place_at_grid(magnifier, 'C4', scale_factor=1.3)
        
        # Red tangent line at the zoomed point
        # Positioned at C4 with scale 1.3 (Issue 34)
        tangent_line = Line(
            start=LEFT*0.8,
            end=RIGHT*0.8,
            color="#FF0000"
        )
        # Rotate to show a slope (conceptually the derivative at that point)
        tangent_line.rotate(25 * DEGREES)
        self.place_at_grid(tangent_line, 'C4', scale_factor=1.3)
        
        self.play(FadeIn(magnifier))
        self.play(Create(tangent_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # In motion, the derivative of position is velocity.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Label for velocity (Issue 32)
        # Positioned at D5 with scale 0.7
        # Using Text for robustness (B008)
        velocity_label = Text("v = ds/dt", color="#FF0000")
        self.place_at_grid(velocity_label, 'D5', scale_factor=0.7)
        
        self.play(Write(velocity_label))
        self.wait(2)
        
        # Final cleanup
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
