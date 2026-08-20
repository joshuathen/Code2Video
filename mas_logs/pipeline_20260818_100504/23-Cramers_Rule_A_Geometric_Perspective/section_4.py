from manim import *
import os

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
            "Determinants generalize to higher-dimensional volumes.",
            "Zero determinant means vectors are collinear.",
            "Cramer's rule fails for singular matrices."
        ]
        self.setup_layout("Visual Synthesis & Application", lecture_lines)
        
        # Representations for visualization
        cube = Cube(fill_opacity=0.5, color=BLUE).scale(0.8)
        self.place_at_grid(cube, 'B4', scale_factor=0.6)
        
        flat_box = Cube(fill_opacity=0.3, color=RED).scale(0.8).stretch(0.1, 1)
        self.place_in_area(flat_box, 'D2', 'D4', scale_factor=0.5)
        
        # Load SVG assets with fallback
        ruler_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg"
        calculator_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg"
        
        ruler = SVGMobject(ruler_path) if os.path.exists(ruler_path) else Square(color=BLUE).scale(0.5)
        calculator = SVGMobject(calculator_path) if os.path.exists(calculator_path) else Square(color=GREEN).scale(0.5)
        
        self.place_at_grid(ruler, 'B1', scale_factor=0.5)
        self.place_at_grid(calculator, 'E1', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"), Create(cube), FadeIn(ruler))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"), Transform(cube, flat_box), FadeIn(calculator))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final result display
        label = Text("det(A) = 0 => Singular", color=YELLOW, font_size=24)
        self.place_at_grid(label, 'B5', scale_factor=0.7)
        self.play(self.lecture[2].animate.set_color("#FFFFFF"), Write(label))
        self.wait(2)
