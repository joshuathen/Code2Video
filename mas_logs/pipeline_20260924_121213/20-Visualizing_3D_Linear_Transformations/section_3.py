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
            "The determinant represents a scaling factor for volume.",
            "Volume changes from V to V' under transformation.",
            "Determinant is the ratio V'/V.",
            "Example: A matrix with determinant 2 doubles volume.",
            "Stretching an object scales its total enclosed volume."
        ]
        self.setup_layout("Core Concept: The Determinant", lecture_lines)
        
        # Define objects
        # Use SVG asset as requested in Issue 18
        cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", fill_opacity=0.5, color=BLUE)
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2], axis_config={"include_tip": True})
        
        # Setup right-side area for 3D visualization (Issue 27)
        self.place_at_grid(axes, 'D2', scale_factor=0.7)
        self.place_at_grid(cube, 'D2', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes), Create(cube))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        stretched_cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", fill_opacity=0.7, color=GREEN)
        # Fix placement per Issue 28
        self.place_at_grid(stretched_cube, 'D4', scale_factor=0.7)
        self.play(Transform(cube, stretched_cube))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        formula = MathTex(r"det(M) = \frac{V'}{V}").set_color(WHITE)
        # Fix placement per Issue 29
        self.place_at_grid(formula, 'B4', scale_factor=0.9)
        self.play(Write(formula))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        # Using asset per Issue 18: scaling the cube
        scale_cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", color=YELLOW)
        self.place_at_grid(scale_cube, 'E5', scale_factor=0.8)
        self.play(FadeIn(scale_cube))
        self.play(scale_cube.animate.scale(1.5))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        self.play(cube.animate.scale(1.2), run_time=2)
        self.wait(1)
